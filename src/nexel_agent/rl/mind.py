"""The LLM layer over a learned body: now and then (every THINK_S, or at once when something happens to it: spoken
to, asked, hit, given something) one small local model is given the facts - what it carries and how it is, the goal
it is on and since when, what each person about did to it, what it wrote down about them before - and answers with
JSON: the next goal for the policy, who with, what item, a line to say, and notes about people (allies it chose,
favours, grudges), kept in memory.json across sessions. Whether to do what someone asks, to refuse, to team up or to
break off is the model's call from those facts; nothing here says what to do when. The policy pursues the goal and
answers danger by itself, at its own pace (rl/live.py)."""

import json
import re
import time
from pathlib import Path

from ..chat import CHAT_STYLE
from megastructure.tiles import RECIPES
from .goals import KINDS, Goal

THINK_S = 20.0             # (the model it asks: rl/voice.py MODEL, the same server as the words)
SALIENT = ("say", "shout", "give", "attack")
GOALS_DOC = ("Goals: gather (an item), craft (an item), give (an item, to a person), ask (a person, for an item), "
             "stay (with a person), help (a person, in a fight), avoid (a person), free (no goal)")


class Mind:
    def __init__(self, name: str, persona: str, model_url: str, http, path: Path):
        self.name, self.persona, self.url, self.http, self.path = name, persona, model_url, http, Path(path)
        self.relations = json.loads(self.path.read_text()) if self.path.exists() else {}     # name: {ally, notes}
        self.seen: dict = {}              # name: recent things they did (this session)
        self.goal, self.goal_at = Goal(), time.monotonic()
        self.last, self.salient, self.busy = 0.0, False, False
        self.to_say: str | None = None
        self.names: dict = {}
        self.deeds = ""                  # what the body did lately (the live body says: rl/live.py)
        self.asks = ""                   # an ask open now, with what it carries of that item
        self.log = self.path.with_name("mind.jsonl")   # each thought: the facts it got, what it answered
        self.voice = None                                 # (rl/voice.py: the one model, chat lines first)
        self.stats = {"thoughts": 0, "said": 0, "errors": 0, "goals": {}}

    def see(self, obs: dict):
        me = self.name
        for e in obs.get("events", []):
            t, who = e.get("type"), e.get("from") or e.get("attacker")
            if t in ("say", "shout") and who and who != me:
                self.note(who, f'said "{str(e.get("text", ""))[:80]}"', salient=self.voice is None)   # (voice.py answers it)
            elif t == "give" and e.get("to") == me and who:
                self.note(who, f"gave you {e.get('items')}")
            elif t == "give" and who == me:
                self.note(e.get("to"), f"you gave them {e.get('items')}", salient=False)
            elif t == "attack" and e.get("target") == me and who:
                self.note(who, f"hit you ({e.get('damage')})")
            elif t == "attack" and who and e.get("target") and who != me:
                self.note(who, f"hit {e.get('target')}", salient=e.get("target") in self.relations)

    def note(self, who, what, salient=True):
        if not who:
            return
        log = self.seen.setdefault(who, [])
        log.append(f"{time.strftime('%H:%M:%S')} {what}")
        del log[:-6]
        self.salient |= salient

    def due(self) -> bool:
        return not self.busy and (self.salient or time.monotonic() - self.last >= THINK_S)

    def facts(self, obs: dict) -> str:
        me = obs["self"]
        inv = ", ".join(f"{k} {v}" for k, v in sorted(me.get("inventory", {}).items(), key=lambda kv: -kv[1])[:10]) or "nothing"
        g = self.goal
        goal = "free" if g.kind == "free" else " ".join(str(p) for p in (g.kind, g.item, g.person and f"(with {g.person})") if p)
        have = me.get("inventory", {})
        close = sorted(((sum(max(0, n - have.get(k, 0)) for k, n in r.inputs.items()), name, r) for name, r in RECIPES.items()
                        if name not in have), key=lambda t: t[0])
        close = [f"{name} (lacks {', '.join(f'{n - have.get(k, 0)} {k}' for k, n in r.inputs.items() if have.get(k, 0) < n)})"
                 for miss, name, r in close if 0 < miss <= 2][:5]
        lines = [f"You: health {me['hp']}/20, food {me['hunger']}/20, carrying {inv}. {obs.get('time', '')}.",
                 (f"Your goal now: {goal}, set {time.monotonic() - self.goal_at:.0f} s ago." if g.kind != "free" else "You have no goal now.")]
        if close:
            lines.append("Things you could make with a little more: " + "; ".join(close) + ".")
        if self.deeds:
            lines.append("What you did lately: " + self.deeds + ".")
        if self.asks:
            lines.append(self.asks + ".")
        people = [e for e in obs.get("entities", []) if e.get("kind") == "agent"]
        self.names = {str(p.get("name", "")).lower(): p.get("name") for p in people} | {n.lower(): n for n in self.relations}
        for p in people:
            n = p.get("name")
            rel = self.relations.get(n, {})
            lines.append(f"{n}: {abs(p['dx']) + abs(p['dy'])} tiles away, health {p.get('hp')}, holding {p.get('holding')}. "
                         f"Your notes: {rel.get('notes', 'none')}{' (your ally)' if rel.get('ally') else ''}. "
                         f"Lately: {'; '.join(self.seen.get(n, [])) or 'nothing'}.")
        if not people:
            lines.append("Nobody is in sight.")
        others = [n for n in self.relations if n not in {p.get("name") for p in people}]
        if others:
            lines.append("People you know, not here: " + "; ".join(f"{n} ({self.relations[n].get('notes', '')})" for n in others[:6]))
        return "\n".join(lines)

    def messages(self, obs: dict) -> list:
        sys_ = (f"You are {self.name}, a player in a survival sandbox game. Your character: {self.persona}\n"
                "Your body plays by itself (moving, gathering, crafting, fighting back); you choose what it is after. "
                f"{GOALS_DOC}. Reply with JSON only: "
                '{"goal": "gather|craft|give|ask|stay|help|avoid|free", "with": "a name you see, or empty", "item": "a game item name, or empty", "say": "a short chat line or empty", '
                '"relation_update": {"PERSON": {"ally": true|false, "note": "what you want to remember about them"}}}. '
                "The line you say, if any: " + CHAT_STYLE)
        return [{"role": "system", "content": sys_}, {"role": "user", "content": self.facts(obs) + "\n/no_think"}]

    async def think(self, obs: dict):
        self.busy, self.salient, self.last = True, False, time.monotonic()
        try:
            if self.voice is not None:
                out, timing = await self.voice.post(self.messages(obs), 100, words=False)
            else:
                r = await self.http.post(self.url, timeout=40, json={"model": "default_model", "messages": self.messages(obs),
                                                                     "max_tokens": 100, "temperature": 0.7})
                out, timing = r.json()["choices"][0]["message"]["content"] or "", {}
            with open(self.log, "a") as f:
                f.write(json.dumps({"t": round(time.time(), 1), "facts": self.facts(obs), "reply": out, **timing}) + "\n")
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
        g = Goal(kind, (str(d.get("item") or "").strip().lower().replace(" ", "_") or None), who)
        if (g.kind, g.item, g.person) != (self.goal.kind, self.goal.item, self.goal.person):
            self.goal, self.goal_at = g, time.monotonic()
            self.stats["goals"][kind] = self.stats["goals"].get(kind, 0) + 1
        say = str(d.get("say") or "").strip()
        self.to_say = say[:120] or None
        for who, upd in (d.get("relation_update") or {}).items():
            who = self.names.get(str(who).lower(), who)
            if isinstance(upd, dict) and who and who.lower() not in ("no_one", "nobody", "none", "someone", "person", "a name you see"):
                rel = self.relations.setdefault(who, {})
                if "ally" in upd:
                    rel["ally"] = bool(upd["ally"])
                if upd.get("note"):
                    rel["notes"] = str(upd["note"])[:200]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.relations, indent=1))
