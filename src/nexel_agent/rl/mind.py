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

from megastructure.tiles import RECIPES
from .facts import Attack, Give, Say, events, needs, people, short
from .goals import KINDS, Goal
from .voice import CHAT_STYLE, THOUGHT_TOKENS, Now

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
        self.last, self.salient, self.busy = 0.0, False, False
        self.to_say: str | None = None
        self.names: dict = {}
        self.asks = ""                   # an ask open now, with what it carries of that item
        self.log = self.path.with_name("mind.jsonl")   # each thought: the facts it got, what it answered
        self.voice = None                                 # (rl/voice.py: the one model, chat lines first)
        self.stats = {"thoughts": 0, "said": 0, "errors": 0, "goals": {}}

    def see(self, obs: dict):
        """Something that happened to it (spoken to - unless the voice answers -, given, hit, an ally hit) thinks at once."""
        me = self.name
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
                 (f"Your goal now: {goal_text(self.goal)}, set {time.monotonic() - self.goal_at:.0f} s ago."
                  if self.goal.kind != "free" else "You have no goal now.")]
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
        g = Goal(kind, item_name(d.get("item")) or None, who)
        if (g.kind, g.item, g.person) != (self.goal.kind, self.goal.item, self.goal.person):
            self.goal, self.goal_at = g, time.monotonic()
            self.stats["goals"][kind] = self.stats["goals"].get(kind, 0) + 1
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
