"""A home dug into the land (#66): a trench stepping down a layer a tile to a door, and behind it a room tunnelled two
cells high, the land over it its roof and round it its walls, as the game digs (nexel_worlds/sim/underground.py:
`dig` at a height takes the cell in front, from one's feet to two over them, and what is over it stays).

    surface  . . . . . . .          the land's top cell (dz 0) is left over the room: its ceiling
    room     # . . . #               floor DEPTH under the top: cells -1 and -2 dug, from a tile dug beside it
    trench   = = D                    each tile a layer lower than the one before it, the last the door's

A cell is dug from the tile beside it, standing on that tile's floor (DEPTH under, for the room). A design as the
architect chooses it: {"underground": true, "shape", "width"/"depth" or "radius", "door", "furniture"}; no storeys,
no roof (the land is both)."""

from .structure import SIDE, Structure

DEPTH = 3                   # the room's floor under the land's top cell: two cells to stand in, the land the third


class Dug(Structure):
    def __init__(self, spec: dict):
        super().__init__({**spec, "storeys": 1, "roof": False, "path": 0})
        self.storeys = [{p: t for p, t in self.storeys[0].items() if t != "wall"}]    # (the land is the walls)
        d = SIDE[self.sides[0]]
        self.trench = [self.doors[0]] + self._line(self.doors[0], d, DEPTH - 1)        # (the door's tile, deepest)
        self.top = self._line(self.trench[-1], d, 1)[0]                                # (where the digging starts)
        self.n, self.stairs, self.path = 1, [], []

    def digs(self) -> list:
        """[(cell, its stand)] in the order dug: the trench from the top down, then the room out from the door."""
        out, prev = [], (*self.top, 0)
        for i, (x, y) in enumerate(reversed(self.trench)):
            depth = i + 1
            out += [((x, y, -c, "dig"), prev) for c in range(depth)]
            prev = (x, y, -depth)
        near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]   # noqa: E731
        seen, frontier = {self.doors[0]}, [self.doors[0]]
        while frontier:
            nxt = []
            for p in frontier:
                for q in sorted(near(p)):
                    if q in self.inside and q not in seen:
                        seen.add(q)
                        nxt.append(q)
                        out += [((q[0], q[1], -c, "dig"), (p[0], p[1], -DEPTH)) for c in (DEPTH - 1, 1)]
            frontier = nxt
        return out

    def cells(self) -> list:
        """The digs, then the door and the furniture on the room's floor."""
        things = sorted(self.storeys[0].items(), key=lambda kv: (kv[1] != "door", kv[0][1], kv[0][0]))
        return [c for c, _ in self.digs()] + [(x, y, -DEPTH, t) for (x, y), t in things]

    def from_(self) -> dict:
        return {c: (s,) for c, s in self.digs()}

    def ways(self) -> dict:
        """A tile dug is stood on once its cells are all dug."""
        by_tile = {}
        for c, _ in self.digs():
            by_tile.setdefault(c[:2], []).append(c)
        return {(x, y, min(c[2] for c in cs) - 1): cs for (x, y), cs in by_tile.items()}

    def stands(self) -> tuple:
        """The room's floor the door leads to, the trench's tiles, and the top where the digging starts."""
        floor = [(x, y, -DEPTH) for x, y in sorted(self.inside) if (x, y) not in self.storeys[0]]
        trench = [(x, y, -(i + 1)) for i, (x, y) in enumerate(reversed(self.trench))]
        return tuple(floor + trench + [(*self.top, 0)]), ()

    def clear(self) -> set:
        """The land it is dug in, open on top and flat: the room, the land round it, the trench and its top."""
        return self.inside | self.ring | set(self.trench) | {self.top}

    def cost(self) -> int:
        return 0                                                # (nothing gathered: the land dug is the pack's)


def design_of(spec: dict) -> Structure:
    """A design drawn: dug into the land when it says so, else built up."""
    from .garden import Garden
    return Dug(spec) if spec.get("underground") else Garden(spec) if spec.get("garden") else Structure(spec)


class DugView:
    """A dug room in a sim world as the judge reads a world, at its floor `z`: land at head height is a wall (stone),
    a tile one stands on is ground, the land over the head a roof."""

    def __init__(self, sim, z):
        self.sim, self.z = sim, z

    def _step(self, x, y):
        """A step up out of the room (the trench's): solid a layer up, room to stand over it."""
        w = self.sim.world
        return w.solid(x, y, self.z + 1) and not w.solid(x, y, self.z + 2) and not w.solid(x, y, self.z + 3)

    def block(self, x, y):
        w = self.sim.world
        return "stone_wall" if w.solid(x, y, self.z + 1) and not self._step(x, y) else w.block(x, y)

    def ground(self, x, y):
        w = self.sim.world
        return w.ground(x, y) if (w.solid(x, y, self.z) and not w.solid(x, y, self.z + 1)) or self._step(x, y) else "water"

    def roof(self, x, y):
        return int(any(self.sim.world.solid(x, y, z) for z in range(self.z + 3, self.z + 6)))

    def level(self, x, y):
        return self.z
