"""Trigger cards (design 3.2): reactions the MODEL writes as data; the body only evaluates them, every look.

    {"when": {"being": "tusker", "within": 4}, "do": ["move_away {id} 10"], "why": "it charges"}
    {"when": {"hit_by": "any"}, "do": ["attack {id}"], "why": "nobody hits me twice"}

`when` holds perception facts only (all must hold): being (a kind or a name, or "any"), within (tiles), hit_by (a
name, a kind, or "any"), hp_below, food_below, night, in_water. `do` is a list of plan commands or verbs, where
{id} {name} {x} {y} stand for the being that set it off. Cards are shared: one file per world
(agents/_cards.json), each with its author; a player uses its own and everyone's. A card whose firing is followed by
a death within 30 s loses trust; at 0 it is no longer used. The model writes them: when it joins (from its persona
and the creatures it sees described), when a new kind of being first comes into view, and after a hit no card
answered.
"""

import json
import re
import time
from pathlib import Path

from .sim.specs import LEGEND
from .plans import _first_object, parse_command

KEYS = {"being", "within", "hit_by", "hp_below", "food_below", "night", "in_water"}
COOLDOWN_S = 3.0
DEATH_S = 30.0


def valid(card) -> dict | None:
    """A card as the model wrote it, checked and normalised (None when it is not one)."""
    if not isinstance(card, dict) or not isinstance(card.get("when"), dict):
        return None
    when = {k: v for k, v in card["when"].items() if k in KEYS}
    do = card.get("do")
    do = [do] if isinstance(do, str) else do
    if not when or not isinstance(do, list) or not do or not all(isinstance(d, str) and d.strip() for d in do):
        return None
    for k in ("within", "hp_below", "food_below"):
        if k in when:
            try:
                when[k] = float(when[k])
            except (TypeError, ValueError):
                return None
    for k in ("being", "hit_by"):
        if k in when:
            when[k] = str(when[k]).lower().removeprefix("a ").strip()
    do = [d.strip()[:80] for d in do[:3] if runs(fill(d.strip(), {"id": 1, "name": "x", "x": 0, "y": 0}, {"x": 0, "y": 0}))]
    if not do:
        return None                                   # (no command in it that runs)
    return {"when": when, "do": do, "why": str(card.get("why", ""))[:120]}


class Cards:
    def __init__(self, path: Path | None, me: str):
        self.path, self.me = path, me
        self.cards: list = json.loads(path.read_text()) if path and path.exists() else []
        self.last: dict = {}                 # (card index, subject id) -> when it last fired
        self.fired: list = []                # (t, card index): for trust after a death

    def save(self):
        if self.path:
            self.path.write_text(json.dumps(self.cards, indent=0))

    def add(self, raw, why_source="") -> list:
        new = []
        for c in raw if isinstance(raw, list) else [raw]:
            v = valid(c)
            if v and not any(x["when"] == v["when"] and x["do"] == v["do"] and x["author"] == self.me for x in self.cards):
                v.update(author=self.me, trust=1, fired=0, source=why_source, t=round(time.time()))
                self.cards.append(v)
                new.append(v)
        if new:
            self.save()
        return new

    def usable(self):
        """Own cards first, then everyone else's; trust 0 is out."""
        return sorted([(i, c) for i, c in enumerate(self.cards) if c.get("trust", 1) > 0],
                      key=lambda ic: ic[1].get("author") != self.me)

    def match(self, obs, hits, now=None) -> tuple | None:
        """(card index, card, commands) of the first card whose `when` holds now, or None. hits: recent
        [(who, being dict or None)] from this look's events."""
        now = now or time.time()
        me = obs["self"]
        for i, c in self.usable():
            w, subject = c["when"], None
            if "hit_by" in w:
                h = next(((who, b) for who, b in hits if w["hit_by"] in ("any", who.lower().removeprefix("a ")) or
                          (b and w["hit_by"] in (b["kind"], str(b.get("name", "")).lower()))), None)
                if not h:
                    continue
                subject = h[1] or {"name": h[0]}
            if "being" in w or "within" in w:
                kind = w.get("being", "any")
                near = [e for e in obs["entities"] if kind in ("any", e["kind"], str(e.get("name", "")).lower())
                        and abs(e["dx"]) + abs(e["dy"]) <= w.get("within", 99)]
                if subject and subject.get("id") is not None:
                    near = [e for e in near if e["id"] == subject["id"]]
                if not near:
                    continue
                subject = subject or min(near, key=lambda e: abs(e["dx"]) + abs(e["dy"]))
            if "hp_below" in w and not me["hp"] < w["hp_below"]:
                continue
            if "food_below" in w and not me["hunger"] < w["food_below"]:
                continue
            if "night" in w and bool(w["night"]) != ("(night)" in str(obs.get("time", ""))):
                continue
            if "in_water" in w and bool(w["in_water"]) != bool(me.get("swimming")):
                continue
            key = (i, (subject or {}).get("id"))
            if now - self.last.get(key, 0) < COOLDOWN_S:
                continue
            self.last[key] = now
            return i, c, [fill(d, subject, me) for d in c["do"]]
        return None

    def fired_now(self, i, now=None):
        self.cards[i]["fired"] = self.cards[i].get("fired", 0) + 1
        self.fired.append((now or time.time(), i))
        self.save()

    def died(self, now=None):
        """A death: every card fired in the last DEATH_S loses a point of trust."""
        now = now or time.time()
        for t, i in self.fired:
            if now - t <= DEATH_S:
                self.cards[i]["trust"] = self.cards[i].get("trust", 1) - 1
        self.save()

    def text(self, k=4) -> str:
        return " | ".join(f"when {c['when']} do {c['do']}" for _, c in self.usable()[:k] if c.get("author") == self.me)


def fill(cmd, subject, me):
    s = subject or {}
    x = me["x"] + s["dx"] if "dx" in s else s.get("x", "")
    y = me["y"] + s["dy"] if "dy" in s else s.get("y", "")
    return (cmd.replace("{id}", str(s.get("id", s.get("name", "")))).replace("{name}", str(s.get("name") or s.get("kind", "")))
            .replace("{x}", str(x)).replace("{y}", str(y)))


def card_request(name, persona, kinds, situation, rules) -> list[dict]:
    """One call: the model writes its own reaction cards (JSON list) for what it may meet."""
    seen = "\n".join(f"- {k}: {LEGEND.get(c, k)}" for k, c in kinds)
    sys_ = (f"You are {name}. Who you are: {persona}\n{rules}\nYou react in real time through reaction cards you write "
            "yourself: when the situation in `when` holds, the commands in `do` run at once, before you think.\n"
            'A card: {"when": {...}, "do": ["<command>", ...], "why": "<short>"}. `when` has only the keys the card '
            "needs, and all of them must hold: being (a kind or a name or \"any\"), within (tiles), hit_by (a name, a "
            "kind or \"any\"), hp_below, food_below, night (true/false), in_water (true/false). Each `do` item is one "
            "whole command, e.g. \"attack {id}\"; {id} {name} {x} {y} are the being that set it off. "
            "Commands: attack ID [with WEAPON], move_away TARGET [N], go_next_to TARGET, eat ITEM, say TEXT, "
            "move_to X Y, place_near ITEM.\nAnswer with a JSON list of 1 to 4 cards, nothing else.")
    return [{"role": "system", "content": sys_},
            {"role": "user", "content": f"{situation}\nBeings you may meet:\n{seen}\nYour cards? /no_think"}]


def parse_cards(text: str) -> list:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    i, j = text.find("["), text.rfind("]")
    if i != -1 and j > i:
        try:
            v = json.loads(text[i:j + 1])
            return v if isinstance(v, list) else []
        except ValueError:
            pass
    o = _first_object(text)
    return [o] if o else []


def runs(cmd) -> bool:
    """A whole command: it parses, and a tool that takes arguments got some ("say" alone is not one)."""
    from .plans import SPECS
    c = parse_command(cmd)
    return bool(c) and (bool(c[1]) or not SPECS[c[0]]["args"])
