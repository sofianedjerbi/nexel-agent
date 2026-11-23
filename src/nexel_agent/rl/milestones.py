"""The achievements read from the world after each decision (rl/achieve.py has the list): where the player has
been (chunks; a cave and the deep, counted once back up in the open), whether it stands in a shelter it built (the game's own `sheltered`), a lit one, a
house (a door and a bed it placed, near), the nights lived through, a day never under half health, a swim across,
a foe that came close and did no harm. What happened, never how."""

import math

from ..gen import CHUNK
from .achieve import HOSTILE

EVADE_TICKS = 100          # a foe within 2 tiles, and 10 s later no blow taken since


class Milestones:
    def __init__(self, sim, a):
        self.sim, self.a = sim, a
        self.chunks: set = set()
        self.nights, self.night = 0, sim.is_night()
        self.day_min, self.full_day = a.hp, False
        self.swim_from = None
        self.near_since, self.last_hit = None, -10 ** 9
        self.been = set()          # cave, depth_5, depth_10: been there, not yet back up in the open

    def update(self, L, hurt: bool, died: bool) -> float:
        sim, a = self.sim, self.a
        u = L.unlock
        r = 0.0
        if hurt:
            self.last_hit = sim.tick
        self.chunks.add((a.x // CHUNK, a.y // CHUNK))
        for n in (4, 9, 16):
            r += u(f"chunks_{n}") if len(self.chunks) >= n else 0
        cave = sim.in_cave(a)
        z = sim.height(a)
        self.been |= {k for k, at in (("enter_cave", cave), ("depth_5", z <= -5), ("depth_10", z <= -10)) if at}
        if self.been and not cave and z >= 0 and not died:       # (a trip down counts once you are back up)
            r += sum(u(k) for k in self.been)
            self.been = set()
        if L.placed and not cave and sim.sheltered(a.x, a.y):
            r += u("shelter")
            if sim.near_block(a, "campfire", 3):
                r += u("light_shelter")
            if {"door", "bed"} <= L.placed and sim.near_block(a, "door", 6) and sim.near_block(a, "bed", 6):
                r += u("house")
        if a.count("axe") and a.count("pickaxe") and a.count("hoe") and (a.count("spear") or a.count("sword")):
            r += u("toolkit")
        if sim.swimming(a):
            self.swim_from = self.swim_from or (a.x, a.y)
        elif self.swim_from is not None:
            if math.dist(self.swim_from, (a.x, a.y)) >= 5 and not died:
                r += u("swim_across")
            self.swim_from = None
        near = any(c.kind in HOSTILE and max(abs(c.x - a.x), abs(c.y - a.y)) <= 2 for c in sim.creatures.values())
        if near and self.near_since is None:
            self.near_since = sim.tick
        if self.near_since is not None:
            if self.last_hit >= self.near_since:
                self.near_since = None
            elif sim.tick - self.near_since >= EVADE_TICKS and not died:
                r += u("evade_foe")
                self.near_since = None
        self.day_min = min(self.day_min, a.hp)
        night = sim.is_night()
        if self.night and not night and not died:                # a dawn
            self.nights += 1
            r += u(("survive_night", "survive_2_nights", "survive_3_nights")[min(self.nights, 3) - 1])
            if self.full_day and self.day_min >= 10:
                r += u("hale_day")
            self.day_min, self.full_day = a.hp, True
        self.night = night
        return r
