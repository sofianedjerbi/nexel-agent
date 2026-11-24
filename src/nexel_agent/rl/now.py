"""What is happening to the player now, in plain facts, for every prompt (the words: rl/voice.py; the planner:
rl/mind.py): health and food and how they went, what it carries, who is near and what they did, what lies on the
ground, what its body did, what it got or lost, what was said, the places it knows. Fed each observation and each
action of its body (rl/live.py)."""

import math
import time
from collections import Counter, deque
from itertools import groupby

from .facts import (HOSTILE, Attack, Craft, Give, Say, compass, counted, dist, events, ground, is_creature, named, pack,
                    people, plain, told)

SIGHTS = ("water", "deep_water", "tree", "rock", "ore", "thicket", "berry_bush", "fruit_tree", "campfire", "workbench",
          "bed", "door", "chest", "well", "cave_floor", "hole", "ladder", "stone_wall", "wood_wall", "fence")
WINDOW_S = 30.0                       # "lately": what it saw in the last 30 s; an ask not met in that time was not met
WORDS = {"N": "north", "S": "south", "E": "east", "W": "west", "noop": "wait"}


def places(obs: dict, signs: dict) -> str:
    """The real places it knows of: signs it has read (where, what they say), its spawn, what is in sight now."""
    me = obs.get("self") or {}
    x, y = me.get("x", 0), me.get("y", 0)
    out = [f'a sign "{t[:40]}" {compass(sx - x, sy - y)}' for (sx, sy), t in list(signs.items())[-3:]]
    sp = me.get("spawn")
    if sp and math.hypot(sp[0] - x, sp[1] - y) > 3:
        out.append(f"your spawn point {compass(sp[0] - x, sp[1] - y)}")
    legend = obs.get("legend") or {}
    seen = {legend.get(ch, "") for row in obs.get("grid") or [] for ch in row}
    sights = [s.replace("_", " ") for s in SIGHTS if s in seen]
    if sights:
        out.append("in sight: " + ", ".join(sights[:6]))
    if me.get("in_cave"):
        out.append("you are in a cave")
    return "; ".join(out) or "none"


def runs(words) -> str:
    """'a, a, b' -> 'a x2, b' (in order)."""
    return ", ".join(w + (f" x{n}" if (n := len(list(g))) > 1 else "") for w, g in groupby(words))


class Now:
    """What is happening to the player now, in plain facts: the last observation and what it saw in the last WINDOW_S.
    Fed each observation (see) and each action of its body (did); read by every prompt (text). It also keeps, per
    person, a record of what happened between them - outcomes only, counted: asked, given, not given, hit, helped -
    in the planner's memory (rl/mind.py relations[name]["record"])."""

    def __init__(self, name: str, items=()):
        self.name, self.obs, self.signs, self.items = name, {}, {}, tuple(items)   # items: what a pack can hold
        self.relations: dict = {}            # name: {ally, record} (rl/mind.py keeps it across sessions)
        self.asks: dict = {}                 # (who asked, whom): (item, t), open for WINDOW_S
        self.fought = -1e9                   # when a creature last hit it
        self.past: deque = deque()           # (t, event) within the window
        self.body: deque = deque()           # (t, hp, food, inventory): the oldest is the window's start
        self.acts: deque = deque(maxlen=6)

    def see(self, obs: dict):
        t, me = time.monotonic(), obs.get("self") or {}
        self.obs = obs
        for s in obs.get("signs", []):
            self.signs[(me.get("x", 0) + s["dx"], me.get("y", 0) + s["dy"])] = str(s.get("text", ""))
        for ev in events(obs):
            self.past.append((t, ev))
            self.outcome(ev, t)
        for (asker, whom), (item, t0) in [kv for kv in self.asks.items() if t - kv[1][1] > WINDOW_S]:
            del self.asks[(asker, whom)]
            self.remember(whom, f"did not give you the {item} you asked for") if asker == self.name else \
                self.remember(asker, f"you did not give them the {item} they asked for")
        self.body.append((t, me.get("hp"), me.get("hunger"), dict(me.get("inventory") or {})))
        while self.past and t - self.past[0][0] > WINDOW_S:
            self.past.popleft()
        while len(self.body) > 1 and t - self.body[1][0] >= WINDOW_S:
            self.body.popleft()

    def remember(self, who, what: str, n=1):
        if who and who != self.name and not is_creature(who):     # (people only)
            rec = self.relations.setdefault(who, {}).setdefault("record", {})
            rec[what] = rec.get(what, 0) + n

    def outcome(self, ev, t: float):
        me = self.name
        match ev:
            case Attack(by=a, target=tgt) if tgt == me:
                self.remember(a, "hit you")
                self.fought = t if is_creature(a) else self.fought
            case Attack(by=a, target=tgt) if a == me:
                self.remember(tgt, "you hit them")
            case Attack(by=a, target=tgt) if is_creature(tgt) and t - self.fought <= WINDOW_S:
                self.remember(a, "fought a creature that was fighting you")
            case Give(src=src, to=to) if me in (src, to):
                for k, n in ev.items.items():
                    self.remember(src if to == me else to, f"gave you {k}" if to == me else f"you gave them {k}", n)
                    if self.asks.get((to, src), ("",))[0] == k:
                        del self.asks[(to, src)]
            case Say(src=src) if src != me and (item := next(iter(named(ev.text, self.items)), None)):
                self.ask(src, me, item, t)

    def ask(self, who, whom, item, t=None):
        """`who` asked `whom` for `item` (its own asks: the policy's ask action, rl/live.py)."""
        whom = whom or next((b.get("name") for b in people(self.obs)), None)
        if who and whom:
            self.asks[(who, whom)] = (item, time.monotonic() if t is None else t)
            self.remember(who if who != self.name else whom, f"asked you for {item}" if who != self.name else f"you asked them for {item}")

    def person(self, name) -> str:
        """What happened between it and `name` so far, and whether it calls them an ally."""
        rel = self.relations.get(name) or {}
        rec = [f"{w} x{n:g}" if n != 1 else w for w, n in (rel.get("record") or {}).items()]
        ally = {True: "your ally", False: "not your ally"}.get(rel.get("ally"))
        return "; ".join(([ally] if ally else []) + rec) or "nothing between you yet"

    def did(self, action: str):
        self.acts.append(" ".join(WORDS.get(w, w) for w in action.split("_")))

    def who(self, name) -> str:
        return "you" if name == self.name else plain(name)

    def event(self, ev) -> str | None:
        """An event of the window in words (rl/facts.py told); chat has its own line, and of crafts only its own count."""
        return None if isinstance(ev, Say) or (isinstance(ev, Craft) and ev.name != self.name) else told(ev, self.who)

    def text(self) -> str:
        obs, now = self.obs, time.monotonic()
        me = obs.get("self") or {}
        t0, hp0, food0, inv0 = self.body[0] if self.body else (now, me.get("hp"), me.get("hunger"), {})
        ago = f"{now - t0:.0f} s ago"
        out = [f"You: health {me.get('hp')}/20" + (f" (was {hp0} {ago})" if hp0 != me.get("hp") else "")
               + f", food {me.get('hunger')}/20" + (f" (was {food0} {ago})" if food0 != me.get("hunger") else "")
               + (f". {obs['time']}." if obs.get("time") else "."), f"You carry {pack(obs)}."]
        beings = sorted(obs.get("entities", []), key=dist)
        near = []
        for kind in dict.fromkeys(b["kind"] for b in beings if b.get("kind") != "agent"):
            d = [round(dist(b)) for b in beings if b["kind"] == kind]
            near.append(f"{len(d)} {kind}{' (foe)' if kind in HOSTILE else ''} {d[0]}" + (f"-{d[-1]}" if d[-1] != d[0] else "")
                        + " tiles away")
        for b in people(obs):
            near.append(f"{b.get('name')} {compass(b['dx'], b['dy'])}, health {b.get('hp')}, holding {b.get('holding') or 'nothing'}, "
                        f"doing: {b.get('doing') or 'idle'} (so far: {self.person(b.get('name'))})")
        near += [f"{counted(items)} on the ground {compass(dx, dy)}" for dx, dy, items in ground(obs)[:5]]
        out.append("Near you: " + ("; ".join(near) if near else "nobody, no creature, nothing on the ground") + ".")
        seen = Counter(p for _, e in self.past if (p := self.event(e)))
        if seen:
            out.append(f"In the last {WINDOW_S:.0f} s: " + "; ".join(p + (f" x{n}" if n > 1 else "") for p, n in seen.items()) + ".")
        if self.acts:
            out.append("Your body's last actions: " + runs(self.acts) + ".")
        inv = me.get("inventory") or {}
        got = {k: n - inv0.get(k, 0) for k, n in inv.items() if n > inv0.get(k, 0)}
        lost = {k: n - inv.get(k, 0) for k, n in inv0.items() if n > inv.get(k, 0)}
        if got or lost:
            out.append(f"Since {ago}: " + "; ".join(p for p in (got and "you got " + counted(got),
                                                                   lost and "you used or lost " + counted(lost)) if p) + ".")
        chat = [told(ev, self.who) for _, ev in self.past if isinstance(ev, Say)]
        if chat:
            out.append("Said near you lately: " + " | ".join(chat[-4:]))
        out.append("Places you know: " + places(obs, self.signs))
        return "\n".join(out)
