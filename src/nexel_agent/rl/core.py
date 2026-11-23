"""The learning env over the in-process sim (the game's own Python: no gymnasium here; rl/vec.py wraps it).

An episode: a freshly generated world (seed), one player at its spawn in the morning, mobs, food and night as in the
game, until the next dawn (or death). A decision is taken when the last action ends (0.2 s at the soonest), when the player is hurt,
or every DECIDE ticks (0.5 s) at the latest; `noop` lets a long action (chopping) go on."""

import math
from pathlib import Path

from ..sim import Simulation
from ..sim.constants import DAY_TICKS
from .achieve import HOSTILE, Ledger
from .encode import ACTIONS, encode, legal, to_command

DECIDE = 5                          # ticks: the longest a decision waits
MIN_TICKS = 2                       # and the shortest (0.2 s: a step), unless hurt
START = 0.3                         # the time of day an episode starts (morning)
EP_TICKS = int(0.75 * DAY_TICKS)    # to the dawn after the first night, and a little more
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
        self.pending, self.running = [], None
        self.night = self.sim.is_night()
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
        if kind in ("step", "jump", "dig", "dig_at") and ACTIONS[i][1]:
            self.facing = ACTIONS[i][1]
        inv0, hp0, deaths0 = dict(a.inventory), a.hp, a.deaths
        act = None
        if cmd is not None and self.running and self.running[0] == cmd and self.running[1].result is None:
            cmd = None                      # the same action again, still under way: it goes on (not started over)
        if cmd is None:
            L.counts["idle"] += a.action is None
        else:
            tool, args = cmd
            target = sim.creatures.get(args.get("target")) if tool == "attack" else None
            act = sim.submit(a, tool, args)
            self.running = (cmd, act)
            self.pending.append((act, tool, args, target.kind if target else ""))
        for t in range(self.decide):
            sim.step()
            if a.deaths > deaths0 or a.hp < hp0:
                break
            if t + 1 >= MIN_TICKS and act is not None and act.result is not None:
                break
        r, made = 0.0, {}
        self.last_ok = 0.0
        for p in [p for p in self.pending if p[0].result is not None]:
            self.pending.remove(p)
            res = p[0].result
            r += L.result(p[1], p[2], res, p[3])
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
        night = sim.is_night()
        if self.night and not night and not died:
            r += L.unlock("survive_night")
        self.night = night
        self.last = i
        self.obs = sim.observe(a)
        truncated = sim.tick - self.t0 >= self.ep_ticks
        info = {}
        if died or truncated:
            L.close_fights(died)
            info = self.summary(died)
        return self.encoded(), r, died, truncated and not died, info

    def summary(self, died: bool) -> dict:
        c = self.ledger.counts
        return {"achievements": sorted(self.ledger.done), "ticks": self.sim.tick - self.t0, "died": died,
                "distinct_crafts": len(self.ledger.crafted), **c, "reactions": dict(self.reactions)}
