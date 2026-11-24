"""The achievements read from the world after each decision (rl/achieve.py has the list): where the player has
been (chunks; a cave and the deep, counted once back up in the open), whether it stands in a shelter it built (the game's own `sheltered`), a lit one, a
house (a door and a bed it placed, near), the nights lived through, a day never under half health, a swim across,
a foe that came close and did no harm. What happened, never how."""

import math

from nexel_worlds.gen import CHUNK
from nexel_worlds.tiles import DIRS
from .achieve import HOSTILE

ROOM_PIECES = ("wood_wall", "stone_wall", "door", "gate", "fence")

# Steps toward a room, paid once an episode each, less than a room all told (shaping, not achievements): the 8 tiles
# round it closed by its own pieces, and the wall pieces it holds at once.
LADDERS = {"fill": {4: 0.2, 6: 0.2, 8: 0.2}, "stock": {4: 0.1, 8: 0.1}}
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
        self.mine, self.paid = set(), set()        # tiles it put a wall, door or gate on; ladder steps paid

    def room(self, most=60):
        """The space round the player closed off by walls, doors and gates (the game's ENCLOSE), small and compact
        (as the game's own rooms): (the tiles inside, the pieces round it), or None."""
        sim, a = self.sim, self.a
        seen, edge, stack = {(a.x, a.y)}, set(), [(a.x, a.y)]
        while stack:
            cx, cy = stack.pop()
            if len(seen) > most:
                return None
            for dx, dy in DIRS.values():
                n = (cx + dx, cy + dy)
                if n in seen or n in edge:
                    continue
                if sim.world.block(*n) in sim.ENCLOSE:
                    edge.add(n)
                elif not sim.stand(*n) and sim.world.block(*n) not in ("bed", "chair", "table", "chest", "workbench", "campfire"):
                    return None                           # (a cliff or a rock is not a wall it built)
                else:
                    seen.add(n)
                    stack.append(n)
        return seen, edge

    def ladder(self, L, name: str, have: int) -> float:
        """The steps of ladder `name` reached at `have`, each paid once an episode (broken and redone: not again)."""
        new = [n for n in LADDERS[name] if have >= n and (name, n) not in self.paid]
        self.paid |= {(name, n) for n in new}
        L.counts.update({f"{name}_{n}": 1 for n in new})
        return sum(LADDERS[name][n] for n in new)

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
            room = self.room() if L.placed & set(ROOM_PIECES) else None
            if room is not None:                          # (walled all round: not just walls beside it)
                inside, edge = room
                r += u("room_built")
                r += u("roofed_room") if sim.world.roof(a.x, a.y) else 0
                r += u("room_with_door") if any(sim.world.block(*p) in ("door", "gate") for p in edge) else 0
                r += u("bed_in_room") if any(sim.world.block(*p) == "bed" for p in inside) else 0
        r += self.ladder(L, "fill", sum((a.x + dx, a.y + dy) in self.mine and sim.world.block(a.x + dx, a.y + dy) in
                                        sim.ENCLOSE for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy))
        r += self.ladder(L, "stock", sum(a.count(w) for w in ("stone_wall", "wood_wall")))
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
