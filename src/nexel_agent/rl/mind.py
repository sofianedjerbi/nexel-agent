"""The LLM layer over a learned body: now and then (every THINK_S, or at once when something happens to it: spoken
to, asked, hit, given something) one small local model is given the facts - what is happening to it now (the same
text the words get: rl/voice.py Now), what it carries, its own standing aims and how far it is from each, the goal it
is on and since when, what happened between it and each person (outcomes only) - and answers with JSON: the next goal
for the policy, who with, what item, a line to say, its aims, who it calls an ally, what someone told it a thing
takes (the game's own recipes it reads itself; a person is who it asks for the rest); aims, allies, the records and
what it was told are kept in memory.json across sessions. Whether to do what someone asks, to refuse, to team up or to
break off is the model's call from those facts; nothing here says what to do when. The policy pursues the goal and
answers danger by itself, at its own pace (rl/live.py)."""

import json
import re
import time
from pathlib import Path

from nexel_worlds.sim.constants import sec
from nexel_worlds.tiles import RECIPES
from .facts import Attack, Give, Say, counted, dist, events, named, needs, people, short
from .actions import BLOCK_NAME
from .agenda import Agenda
from .goals import KINDS, Goal, account, judge
from .pursue import DROPS
from . import home
from .now import Now
from .voice import CHAT_STYLE, THOUGHT_TOKENS

PROMISE = sec(120)         # a promise to hand something over is kept within this, or broken
THINK_S = 20.0             # (the model it asks: rl/voice.py MODEL, the same server as the words)
HOME = "home"               # an aim that is no item: its home, what it takes its lacks (rl/home.py)
GOALS_DOC = ("Goals: gather (an item), craft (an item), give (an item, to a person), ask (a person, for an item), "
             "stay (with a person), help (a person, in a fight), avoid (a person), build (your home: the body lays "
             "it, walls, door, fire, bed, roof, from what you carry, making what it lacks), free (no goal). Change it "
             "whenever you see fit: a goal you leave before it is done is kept, set aside, in your facts")
BUILD = "build your home"     # (the build goal: no goal kind the body knows, the mind's own - steer)
BUILT = 5                    # a home built: the judge's v1 at 5 of its 6 checks or more (rl/judge.py CHECKS)
LACKING = {"closed": "walls all round", "door": "a door", "roofed": "a roof", "lit": "a light", "bed": "a bed",
           "room_enough": "room inside"}


def item_name(s) -> str:
    """An item as the game spells it ("Stone Block" is stone_block)."""
    return str(s or "").strip().lower().replace(" ", "_")


def goal_text(g: Goal) -> str:
    return "" if g.kind == "free" else " ".join(str(p) for p in (g.kind, g.item, g.person and f"(with {g.person})") if p)


class Mind:
    def __init__(self, name: str, persona: str, model_url: str, http, path: Path, now: Now | None = None):
        self.name, self.persona, self.url, self.http, self.path = name, persona, model_url, http, Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)      # (its notes, thoughts and timings go there)
        mem = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.aims = list(mem.get("aims", []))                  # what it wants to own or build, its own choice
        self.agenda = Agenda(mem.get("agenda", []))            # goals set aside, kept across sessions too
        self.told = dict(mem.get("told", {}))                  # item: what a person told it that takes, in their words
        self.now = now or Now(name)       # (what is happening to it: fed by the live body; it keeps the records)
        self.relations = self.now.relations                  # name: {ally, record}
        self.relations.update(mem.get("people", {k: v for k, v in mem.items() if k not in ("aims", "told", "home")}))
        self.home = tuple(mem["home"]) if mem.get("home") else None   # where its home is (the body chose it: rl/home.py)
        self.social, self.home_back, self.home_read = None, False, None   # (the body's memory, live.py hands it: its home handed back once)
        self.goal, self.goal_at = Goal(), time.monotonic()
        self.tick, self.pack, self.ended = 0, {}, ""     # (the game's tick and the pack, last seen; how the last goal went)
        self.where: dict = {}             # each person: where last seen (rl/social.py SocialMemory.where, live.py hands it)
        self.last, self.salient, self.busy = 0.0, False, False
        self.to_say: str | None = None
        self.tried: dict = {}                # goal text: [set, done] this session
        self.building, self.build_words = False, None   # its home being laid, the planner's call (steer); what, in words
        self.build_end = ""                  # why the building ended, till the next facts grade the home and tell it
        self.names: dict = {}
        self.asks = ""                   # an ask open now, with what it carries of that item
        self.log = self.path.with_name("mind.jsonl")   # each thought: the facts it got, what it answered
        self.voice = None                                 # (rl/voice.py: the one model, chat lines first)
        self.stats = {"thoughts": 0, "said": 0, "errors": 0, "goals": {}}

    def keep_home(self):
        """The home the body chose (rl/build.py social.home), kept: at the start, the one it had last session is handed
        back to a body that has none; after that it is the body's, kept as it is (dropped when the body gives it up)."""
        s = self.social
        if s is None:
            return
        if not self.home_back:
            self.home_back = True
            if s.home is None and self.home:
                s.home = self.home
        self.home = tuple(s.home) if s.home else None

    def see(self, obs: dict):
        """Something that happened to it (spoken to - unless the voice answers -, given, hit, an ally hit) thinks at once."""
        self.keep_home()
        me, self.tick, self.pack = self.name, obs.get("tick", self.tick), dict((obs.get("self") or {}).get("inventory") or {})
        for ev in events(obs):
            match ev:
                case Say(src=who):
                    mine = self.voice is None
                case Give(src=who, to=to):
                    mine = to == me
                case Attack(by=who, target=tgt):
                    mine = tgt == me or tgt in self.relations
                case _:
                    continue
            self.salient |= mine and bool(who and who != me)

    def track(self, obs: dict, results: list, got_from: dict):
        """Judges the goal as the sim does (rl/goals.py judge, with its timeout). When it ends, done or not, the goal is
        free again, as in training, and the planner thinks at once, told how it went."""
        g, tick = self.goal, obs.get("tick", self.tick)
        if g.kind == "free":
            return
        p = next((b for b in people(obs) if b.get("name") == g.person), None)
        done = judge(g, tick, lambda k: self.pack.get(k, 0), results, got_from, dist(p) if p else 99.0, p and p.get("id"))
        if done or tick >= g.until:
            self.end(g, account(g, tick, "done" if done else "timed out, not done"), kept=bool(done))

    def commit(self, to: str, item: str, n: int = 1):
        """Its own line promised `to` the item (rl/acts.py): the promise is its goal now, a give with a deadline. Kept
        or broken, it goes on the record with that person (rl/now.py), as their gifts and blows do."""
        if self.goal.promise:
            self.end(self.goal, "set aside for another promise", kept=False)
        g = Goal("give", item, to, self.tick, self.pack.get(item, 0), until=self.tick + PROMISE)
        g.seen, g.promise = self.where.get(to), True
        self.goal, self.goal_at = g, time.monotonic()
        self.stats["promises"] = self.stats.get("promises", 0) + 1

    async def speak(self, voice, line: str):
        """Its own line, through the one speaker. Said to the person of an ask goal, it is the ask: on the record, as
        the body's own asks are (live trial 4: 25 lines asking Mira for wood went unrecorded, the goal read 'you have
        not asked yet', and it asked again 96 plans running)."""
        g = self.goal
        r = await voice.speak(line, g.person)
        if r is not None and g.kind == "ask" and g.item and g.person:
            g.asked += 1
            self.now.ask(self.name, g.person, g.item)
        return r

    def steer(self, i: int, mask, hurt: float) -> int:
        """The body's action, or build while the mind is building its home (the planner's call: goal 'build'): the
        learned body has not learned to finish a hut (r51-r65), the build verb does it (rl/build.py). Hurt, the body
        acts as it learned (a fight, a flight); build with nothing to lay ends the building, told in the next facts."""
        from .actions import ACTIONS
        b = next(k for k, a in enumerate(ACTIONS) if a[0] == "build")
        if not self.building or hurt > 0:
            return i
        if mask is not None and not mask[b]:
            self.stop_building("nothing more it can lay or make here now")
            return i
        return b

    def stop_building(self, why: str):
        """The building ends: told and tallied in the next facts, with how the home stands then (built)."""
        self.building, self.build_end, self.ended = False, why, f"{BUILD}: {why}"

    def stands(self) -> tuple:
        """How its home stands, from the home's reading (rl/home.py reading: the judge's v1 on the observation, as the
        home facts line says it): ("it stands at 4/6, lacking a roof", built: 5 of 6 or more)."""
        r = self.home_read
        if not self.home:
            return "you have no home yet", False
        if not r or r.get("far"):
            return "your home is too far to see how it stands", False
        v = r["verdict"]
        lack = [LACKING.get(k, k.replace("_", " ")) for k in v["failed"]]
        return f"it stands at {v['grade']}/{v['of']}" + (f", lacking {', '.join(lack)}" if lack else ", whole"), v["grade"] >= BUILT

    def built(self):
        """The building that ended since the last facts, graded: done when the home stands at 5 of 6 or more."""
        if not self.build_end:
            return
        how, done = self.stands()
        self.ended = f"{BUILD}: {how}; {self.build_end}"
        tally = self.tried.setdefault(BUILD, [0, 0])
        tally[0] += 1
        tally[1] += done
        self.build_end = ""

    def end(self, g: Goal, how: str, kept: bool = False):
        if g.promise:
            self.now.remember(g.person, "you kept a promise to them" if kept else "you broke a promise to them")
            how += "; it was a promise"
        self.ended = f"{goal_text(g)}: {how}"
        if how.startswith("done"):
            self.agenda.taken_up(g)
        if g.kind != "free":                                    # (how each goal it set went, this session)
            tally = self.tried.setdefault(goal_text(g), [0, 0])
            tally[0] += 1
            tally[1] += how.startswith("done")
        self.goal, self.goal_at, self.salient = Goal(start=self.tick), time.monotonic(), True

    def due(self) -> bool:
        return not self.busy and (self.salient or time.monotonic() - self.last >= THINK_S)

    def home_needs(self, obs: dict) -> str:
        """What its home still takes, for the aim "home": its lacks beyond the pack, or that it has not chosen one."""
        if not self.home:
            return "you have no home yet: find a flat spot and build"
        lack = home.lacks(obs, self.home, self.social)
        return f"it lacks {counted(lack)}" if lack else "you have all it takes"

    def facts(self, obs: dict) -> str:
        """What is happening now (rl/voice.py Now: the same text the words get, the pack in it), then the planner's own:
        its aims and what each takes (the game's recipes: facts.needs), the goal, what it could make, what people told
        it, an open ask, the people it knows who are not here."""
        me = obs["self"]
        have, known = me.get("inventory", {}), self.now.items
        self.home_read = home.reading(obs, self.home, self.social) if self.home else None    # (what it is told of its home: the brain view shows it)
        self.built()
        close = sorted((sum(short(r, have).values()), name) for name, r in RECIPES.items() if name not in have)
        close = [f"{name} ({needs(name, obs)})" for miss, name in close if 0 < miss <= 2][:5]
        lines = [self.now.text(),
                 ("Your aims: " + "; ".join(f"{a} ({self.home_needs(obs) if a == HOME else needs(a, obs, known)})" for a in self.aims) + "."
                  if self.aims
                  else "You have set yourself no aims yet."),
                 (f"Your goal now: {BUILD}: {self.stands()[0]}." if self.building else
                  f"Your goal now: {goal_text(self.goal)}: {account(self.goal, self.tick)}."
                  if self.goal.kind != "free" else "You have no goal now.")]
        lacks = self.lacks(have, {p.get("name") for p in people(obs)}, obs)
        if lacks:
            lines.append("What your aims lack, and who had it: " + "; ".join(lacks) + ".")
        if self.home_read:
            lines.append(home.told(obs, r=self.home_read))
        if day := home.day(obs, self.home, self.social):
            lines.append(day)
        if self.ended:
            lines.append(f"Your last goal: {self.ended}.")
        if aside := self.agenda.line():
            lines.append(aside)
        if self.tried:
            lines.append("Your goals so far, and how often done: " + "; ".join(
                f"{t}: {d} of {n}" for t, (n, d) in sorted(self.tried.items(), key=lambda kv: -kv[1][0])[:5]) + ".")
        if me.get("craftable"):
            lines.append("Your pack has enough to make: " + ", ".join(list(me["craftable"])[:8]) + ".")
        if close:
            lines.append("Things you could make with a little more: " + "; ".join(close) + ".")
        if self.told:
            lines.append("What people told you: " + "; ".join(f"{k} ({v})" for k, v in list(self.told.items())[-6:]) + ".")
        if self.asks:
            lines.append(self.asks + ".")
        here = {p.get("name") for p in people(obs)}
        self.names = {str(n).lower(): n for n in here | set(self.relations)}
        others = [n for n in self.relations if n not in here]
        if others:
            lines.append("People you know, not here: " + "; ".join(f"{n} ({self.now.person(n)})" for n in others[:6]))
        return "\n".join(lines)

    def lacks(self, have: dict, here: set, obs: dict | None = None) -> list:
        """Each input its aims lack (the game's recipes), with the people it knows had some (rl/now.py who_has: seen
        with it or gave it some), how far it trusts each and who owes whom, who of them is here, and what in sight gives
        it (the game's drops: a tree, wood). Facts only: whether to ask, trade or go and get it is the model's call."""
        out, seen = [], set("".join((obs or {}).get("grid") or []))
        for item, n in {k: n for a in self.aims if a in RECIPES for k, n in short(RECIPES[a], have).items()}.items():
            had = [", ".join([p + (" (here)" if p in here else "")] + self.now.standing(p)) for p in self.now.who_has(item)[:2]]
            src = sorted({BLOCK_NAME[c].replace("_", " ") for c in DROPS.get(item, ()) & seen})
            out.append(f"{n} {item} (" + ("; ".join(had) if had else "nobody you know had any")
                       + (f"; in sight, {' or '.join(src)} gives it" if src else "") + ")")
        return out[:4]

    def messages(self, obs: dict) -> list:
        sys_ = (f"You are {self.name}, a player in a survival sandbox game. Your character: {self.persona}\n"
                "Your body plays by itself (moving, eating, fighting back) and goes after the goal you set: gather sends "
                "it to get an item from the land (it knows what gives what: a tree wood, a rock stone), craft to make one "
                "(it fetches what the recipe takes), ask to ask a person for one, give to hand one over. "
                f"{GOALS_DOC}. The other players know the game too: what a thing takes, where it is found; your line can ask them. "
                "Reply with JSON only: "
                '{"why": "one sentence first: what you want most now, and the quickest way to it from these facts", '
                '"aims": ["1 to 3 items you want to own or build, from your character (kept: repeat them, or change them)"], '
                '"goal": "gather|craft|give|ask|stay|help|avoid|build|free", "with": "a name you see, or empty", "item": "a game item name, or empty", "build": "with build: the build you want, in your own words (what, how big, its roof, of what), or empty", "say": "a short chat line or empty", '
                '"relation_update": {"PERSON": {"ally": true|false}}, '
                '"learned": {"ITEM": {"by": "who told you", "what": "what they said it takes or where it is found, short"}}}. '
                "The line you say, if any: " + CHAT_STYLE)
        return [{"role": "system", "content": sys_}, {"role": "user", "content": self.facts(obs) + "\n/no_think"}]

    async def think(self, obs: dict):
        self.busy, self.salient, self.last = True, False, time.monotonic()
        try:
            msgs = self.messages(obs)
            if self.voice is not None:
                out, timing = await self.voice.post(msgs, THOUGHT_TOKENS, words=False)
            else:
                r = await self.http.post(self.url, timeout=40, json={"model": "default_model", "messages": msgs,
                                                                     "max_tokens": THOUGHT_TOKENS, "temperature": 0.7})
                out, timing = r.json()["choices"][0]["message"]["content"] or "", {}
            with open(self.log, "a") as f:
                f.write(json.dumps({"t": round(time.time(), 1), "facts": msgs[1]["content"].removesuffix("\n/no_think"), "reply": out, **timing}) + "\n")
            self.apply(json.loads(re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", out, flags=re.S), re.S).group(0)))
            self.stats["thoughts"] += 1
        except Exception:
            self.stats["errors"] += 1
        finally:
            self.busy = False

    def apply(self, d: dict):
        kind = str(d.get("goal", "free")).split()[0].lower() if d.get("goal") else "free"
        if self.building and kind != "build":
            self.stop_building("you set another goal")
        self.building = kind == "build"                      # (the home: the body lays it to the end - steer)
        self.build_words = str(d.get("build") or "").strip()[:240] or None if self.building else None   # (architect's brief)
        kind = kind if kind in KINDS else "free"
        who = str(d.get("with") or "").strip()
        who = self.names.get(who.lower(), who) or None               # (the name as the game spells it)
        item = item_name(d.get("item")) or None
        g = Goal(kind, item, who, self.tick, self.pack.get(item, 0) if item else 0)
        g.seen = self.where.get(who) if who else None
        if (g.kind, g.item, g.person) != (self.goal.kind, self.goal.item, self.goal.person):
            if self.goal.promise:                               # (the planner's call: it may break its word)
                self.end(self.goal, "broken: you set another goal")
            else:                                                # (not forgotten: set aside, rl/agenda.py)
                self.agenda.set_aside(self.goal, account(self.goal, self.tick))
            self.agenda.taken_up(g)
            self.goal, self.goal_at = g, time.monotonic()
            self.stats["goals"][kind] = self.stats["goals"].get(kind, 0) + 1
            if item and self.now.items and item not in self.now.items:
                self.end(g, f"there is no item called {item} in the game")
        say = str(d.get("say") or "").strip()
        self.to_say = say[:120] or None
        if isinstance(d.get("aims"), list) and d["aims"]:
            said = [str(a) for a in d["aims"][:3]]               # ("a house with a door and bed": door, bed - the
            aims = [k for a in said for k in named(a, [*self.now.items, HOME])] if self.now.items else \
                [HOME if a.strip().lower() == HOME else item_name(a) for a in said]
            self.aims = list(dict.fromkeys(a for a in aims if a))[:3]   # game's items it names; a phrase is no item)
        for who, upd in (d.get("relation_update") or {}).items():
            who = self.names.get(str(who).lower())              # (people it has seen only)
            if isinstance(upd, dict) and who and "ally" in upd:
                self.relations.setdefault(who, {})["ally"] = bool(upd["ally"])
        for item, t in (d.get("learned") or {}).items():
            by = self.names.get(str(t.get("by") or "").lower()) if isinstance(t, dict) else None   # (people it has seen only)
            if by and by != self.name and item_name(item) and str(t.get("what") or "").strip():
                self.told[item_name(item)] = f'{by}: "{str(t["what"]).strip()[:80]}"'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"aims": self.aims, "agenda": self.agenda.saved(), "people": self.relations, "told": self.told,
                                         **({"home": list(self.home)} if self.home else {})}, indent=1))
