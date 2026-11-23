"""The learning env over the in-process sim (the game's own Python: no gymnasium here; rl/vec.py wraps it).

An episode: a freshly generated world (seed), one player at its spawn in the morning, mobs, food and night as in the
game, until the third dawn (or death). An action runs till it ends (0.2 s at the soonest); the next decision comes then,
or at once when the player is hurt or someone comes into view, or after WAIT_MAX (5 s) at the latest. `noop` lets what
it was doing go on, or, idle, waits 0.5 s."""

import math
from pathlib import Path

from megastructure.sim import Simulation
from megastructure.sim.constants import DAY_TICKS, VIEW_R
from .achieve import HOSTILE, Ledger
from .encode import ACTIONS, encode, legal, to_command
from .milestones import Milestones


def split(cmd) -> list:
    """The tool calls of one action: itself; or, climb, the game's own way up a step: the block over your head dug
    (if any), the land in front at head height (if any), then a jump that way."""
    tool, args = cmd
    if tool == "climb":
        return [("dig", {"dir": "up"}), ("dig", {"dir": args["dir"], "z": args["z"]}), ("jump", {"dir": args["dir"]})]
    return [cmd]

DECIDE = 5                          # ticks: an idle noop waits this long
MIN_TICKS = 2                       # the shortest a decision waits (0.2 s: a step), unless hurt
WAIT_MAX = 50                       # and the longest (a chop by hand takes ~5 s)
START = 0.3                         # the time of day an episode starts (morning)
EP_TICKS = int(2.75 * DAY_TICKS)    # three days and nights: to the third dawn, and a little more
HURT = 0.1                          # reward per half heart lost


class SimEnv:
    def __init__(self, ep_ticks: int = EP_TICKS, decide: int = DECIDE):
        self.ep_ticks, self.decide = ep_ticks, decide
        self.sim = None

    def reset(self, seed: int):
        self.sim = Simulation(Path(":memory:"), seed=seed)
        self.a = self.sim.login("Learner", "rl-token")
        self.sim.tick = self.sim.tick // DAY_TICKS * DAY_TICKS + int(START * DAY_TICKS)
        self.t0 = self.sim.tick
        self.ledger = Ledger()
        self.facing, self.last, self.last_ok, self.hurt = "S", 0, 0.0, 0.0
        self.pending, self.running, self.then = [], None, None
        self.marks = Milestones(self.sim, self.a)
        self.just_hit = False
        self.reactions = {}
        self.obs = self.sim.observe(self.a)
        return self.encoded()

    def encoded(self):
        return {**encode(self.obs, self.facing, self.last, self.last_ok, self.hurt), "mask": legal(self.obs)}

    def step(self, i: int):
        sim, a, L = self.sim, self.a, self.ledger
        cmd = to_command(i, self.obs, self.facing)
        kind = ACTIONS[i][0]
        if self.just_hit:
            self.reactions[kind] = self.reactions.get(kind, 0) + 1
            self.just_hit = False
        if kind in ("step", "jump", "dig", "dig_at", "climb") and ACTIONS[i][1]:
            self.facing = ACTIONS[i][1]
        inv0, hp0, deaths0 = dict(a.inventory), a.hp, a.deaths
        act = None
        if cmd is not None and self.running and self.running[0] == cmd and self.running[1].result is None:
            cmd = None                      # the same action again, still under way: it goes on (not started over)
        if cmd is None:
            L.counts["idle"] += a.action is None
        else:
            first, *then = split(cmd)
            act = self.submit(cmd, *first, then=then)
        # (When the next decision comes, only: being hurt or someone coming into view never chooses what to do; the
        # policy does, and it may well go on with the same action.)
        seen = self.in_view()
        for t in range(WAIT_MAX):
            sim.step()
            if a.deaths > deaths0 or a.hp < hp0:
                break                                   # hurt: decide now
            busy = a.action is not None or a.aside is not None
            if self.then and self.running[1].result is not None:
                act = self.submit(self.running[0], *self.then[0], then=self.then[1:])   # (climb: the next step)
                continue
            if t + 1 >= MIN_TICKS and (act.result is not None if act is not None else not busy and t + 1 >= self.decide):
                break                                   # what it was doing is done (or, idle, 0.5 s went by)
            if t % 5 == 4 and self.in_view() - seen:
                break                                   # someone came into view
        r, made = 0.0, {}
        self.last_ok = 0.0
        for p in [p for p in self.pending if p[0].result is not None]:
            self.pending.remove(p)
            res = p[0].result
            r += L.result(p[1], p[2], res, p[3], a.hp)
            if p[1] == "craft" and res.get("ok"):
                for k, n in (res.get("made") or {}).items():
                    made[k] = made.get(k, 0) + n
            if p[0] is act:
                self.last_ok = 1.0 if res.get("ok") else -1.0 if res.get("status") != "interrupted" else 0.0
        died = a.deaths > deaths0
        r += L.gained(inv0, a.inventory if not died else {}, made)
        lost = hp0 if died else max(0.0, hp0 - a.hp)
        self.hurt = lost
        if lost > 0:
            r -= HURT * lost
            L.counts["hits"] += 1
            self.just_hit = True
            foe = min((c for c in sim.creatures.values() if c.kind in HOSTILE and math.dist((c.x, c.y), (a.x, a.y)) <= 5),
                      key=lambda c: math.dist((c.x, c.y), (a.x, a.y)), default=None)
            if foe is not None:
                L.foes[foe.id] = foe.kind
        r += self.marks.update(L, lost > 0, died)
        self.last = i
        self.obs = sim.observe(a)
        truncated = sim.tick - self.t0 >= self.ep_ticks
        info = {}
        if died or truncated:
            L.close_fights(died)
            info = self.summary(died)
        return self.encoded(), r, died, truncated and not died, info

    def submit(self, cmd, tool, args, then=None):
        target = self.sim.creatures.get(args.get("target")) if tool == "attack" else None
        act = self.sim.submit(self.a, tool, args)
        self.running = (cmd, act)
        self.then = then
        self.pending.append((act, tool, args, target.kind if target else ""))
        return act

    def in_view(self) -> set:
        a, R = self.a, VIEW_R
        return {c.id for c in self.sim.creatures.values() if abs(c.x - a.x) <= R and abs(c.y - a.y) <= R}

    def summary(self, died: bool) -> dict:
        c = self.ledger.counts
        return {"achievements": sorted(self.ledger.done), "ticks": self.sim.tick - self.t0, "died": died,
                "distinct_crafts": len(self.ledger.crafted), "distinct_places": len(self.ledger.placed),
                "nights": self.marks.nights, "chunks": len(self.marks.chunks), **c, "reactions": dict(self.reactions)}
