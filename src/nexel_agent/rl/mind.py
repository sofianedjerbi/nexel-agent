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
from .facts import Attack, Give, Say, dist, events, needs, people, short
from .goals import KINDS, Goal, account, judge
from .now import Now
from .voice import CHAT_STYLE, THOUGHT_TOKENS

PROMISE = sec(120)         # a promise to hand something over is kept within this, or broken
THINK_S = 20.0             # (the model it asks: rl/voice.py MODEL, the same server as the words)
GOALS_DOC = ("Goals: gather (an item), craft (an item), give (an item, to a person), ask (a person, for an item), "
             "stay (with a person), help (a person, in a fight), avoid (a person), free (no goal)")


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
        self.told = dict(mem.get("told", {}))                  # item: what a person told it that takes, in their words
        self.now = now or Now(name)       # (what is happening to it: fed by the live body; it keeps the records)
        self.relations = self.now.relations                  # name: {ally, record}
        self.relations.update(mem.get("people", {k: v for k, v in mem.items() if k not in ("aims", "told")}))
        self.goal, self.goal_at = Goal(), time.monotonic()
        self.tick, self.pack, self.ended = 0, {}, ""     # (the game's tick and the pack, last seen; how the last goal went)
        self.where: dict = {}             # each person: where last seen (rl/social.py SocialMemory.where, live.py hands it)
        self.last, self.salient, self.busy = 0.0, False, False
        self.to_say: str | None = None
        self.names: dict = {}
        self.asks = ""                   # an ask open now, with what it carries of that item
        self.log = self.path.with_name("mind.jsonl")   # each thought: the facts it got, what it answered
        self.voice = None                                 # (rl/voice.py: the one model, chat lines first)
        self.stats = {"thoughts": 0, "said": 0, "errors": 0, "goals": {}}

    def see(self, obs: dict):
        """Something that happened to it (spoken to - unless the voice answers -, given, hit, an ally hit) thinks at once."""
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

    def end(self, g: Goal, how: str, kept: bool = False):
        if g.promise:
            self.now.remember(g.person, "you kept a promise to them" if kept else "you broke a promise to them")
            how += "; it was a promise"
        self.ended = f"{goal_text(g)}: {how}"
        self.goal, self.goal_at, self.salient = Goal(start=self.tick), time.monotonic(), True

    def due(self) -> bool:
        return not self.busy and (self.salient or time.monotonic() - self.last >= THINK_S)

    def facts(self, obs: dict) -> str:
        """What is happening now (rl/voice.py Now: the same text the words get, the pack in it), then the planner's own:
        its aims and what each takes (the game's recipes: facts.needs), the goal, what it could make, what people told
        it, an open ask, the people it knows who are not here."""
        me = obs["self"]
        have, known = me.get("inventory", {}), self.now.items
        close = sorted((sum(short(r, have).values()), name) for name, r in RECIPES.items() if name not in have)
        close = [f"{name} ({needs(name, obs)})" for miss, name in close if 0 < miss <= 2][:5]
        lines = [self.now.text(),
                 ("Your aims: " + "; ".join(f"{a} ({needs(a, obs, known)})" for a in self.aims) + "." if self.aims
                  else "You have set yourself no aims yet."),
                 (f"Your goal now: {goal_text(self.goal)}: {account(self.goal, self.tick)}."
                  if self.goal.kind != "free" else "You have no goal now.")]
        lacks = self.lacks(have, {p.get("name") for p in people(obs)})
        if lacks:
            lines.append("What your aims lack, and who had it: " + "; ".join(lacks) + ".")
        if self.ended:
            lines.append(f"Your last goal: {self.ended}.")
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

    def lacks(self, have: dict, here: set) -> list:
        """Each input its aims lack (the game's recipes), with the people it knows had some (rl/now.py who_has: seen
        with it or gave it some), how far it trusts each and who owes whom, and who of them is here. Facts only:
        whether to ask, trade or go and get it is the model's call."""
        out = []
        for item, n in {k: n for a in self.aims if a in RECIPES for k, n in short(RECIPES[a], have).items()}.items():
            had = [", ".join([p + (" (here)" if p in here else "")] + self.now.standing(p)) for p in self.now.who_has(item)[:2]]
            out.append(f"{n} {item} (" + ("; ".join(had) if had else "nobody you know had any") + ")")
        return out[:4]

    def messages(self, obs: dict) -> list:
        sys_ = (f"You are {self.name}, a player in a survival sandbox game. Your character: {self.persona}\n"
                "Your body plays by itself (moving, gathering, crafting, fighting back); you choose what it is after. "
                f"{GOALS_DOC}. The other players know the game too: what a thing takes, where it is found; your line can ask them. "
                "Reply with JSON only: "
                '{"aims": ["1 to 3 items you want to own or build, from your character (kept: repeat them, or change them)"], '
                '"goal": "gather|craft|give|ask|stay|help|avoid|free", "with": "a name you see, or empty", "item": "a game item name, or empty", "say": "a short chat line or empty", '
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
        kind = kind if kind in KINDS else "free"
        who = str(d.get("with") or "").strip()
        who = self.names.get(who.lower(), who) or None               # (the name as the game spells it)
        item = item_name(d.get("item")) or None
        g = Goal(kind, item, who, self.tick, self.pack.get(item, 0) if item else 0)
        g.seen = self.where.get(who) if who else None
        if (g.kind, g.item, g.person) != (self.goal.kind, self.goal.item, self.goal.person):
            if self.goal.promise:                               # (the planner's call: it may break its word)
                self.end(self.goal, "broken: you set another goal")
            self.goal, self.goal_at = g, time.monotonic()
            self.stats["goals"][kind] = self.stats["goals"].get(kind, 0) + 1
            if item and self.now.items and item not in self.now.items:
                self.end(g, f"there is no item called {item} in the game")
        say = str(d.get("say") or "").strip()
        self.to_say = say[:120] or None
        if isinstance(d.get("aims"), list) and d["aims"]:
            self.aims = [item_name(a) for a in d["aims"][:3] if item_name(a)]
        for who, upd in (d.get("relation_update") or {}).items():
            who = self.names.get(str(who).lower())              # (people it has seen only)
            if isinstance(upd, dict) and who and "ally" in upd:
                self.relations.setdefault(who, {})["ally"] = bool(upd["ally"])
        for item, t in (d.get("learned") or {}).items():
            by = self.names.get(str(t.get("by") or "").lower()) if isinstance(t, dict) else None   # (people it has seen only)
            if by and by != self.name and item_name(item) and str(t.get("what") or "").strip():
                self.told[item_name(item)] = f'{by}: "{str(t["what"]).strip()[:80]}"'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"aims": self.aims, "people": self.relations, "told": self.told}, indent=1))
