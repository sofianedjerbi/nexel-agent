"""What the build verb lays (rl/build.py, #66): a plan's cells (dx, dy, dz, piece) in the order laid, the tiles it is
laid from (dx, dy, dz: dz the height one stands at, over the home's land), spare ones used when they are open ground,
the tiles that must be open for it, a flight of stairs laid from beside it, and the way up to each storey (the stairs'
top tile: a stand up there is used once that flight is laid). The hut (rl/build.py PLAN), a grid the architect drew
(rl/blueprint.py) and a build of storeys (rl/structure.py) are each one."""

from nexel_worlds.sim.bodies import in_reach, reach_of
from nexel_worlds.sim.storeys import STOREY

R = reach_of("use")                 # (the game's reach: its table's and a tile of lead, height counted)
BLOCKS = ("plank", "roof", "stone_block", "dirt")      # what goes in a cell (stone and earth: land in a cell, up a
#                                                       storey only); the rest stands on the land or on a block
SHELL = ("wall", "door", "fence")


class Plan:
    def __init__(self, cells, stands, spare=(), clear=None, from_=None, ways=None):
        self.cells, self.stands, self.spare = list(cells), tuple(stands), tuple(spare)
        self.from_, self.ways = dict(from_ or {}), dict(ways or {})   # (a stand: the cells that open the way to it)
        ground = [(x, y) for x, y, z, p in self.cells if z == 0] + [(x, y) for x, y, z in self.stands if z == 0]
        self.clear = set(clear) if clear is not None else set(ground)
        self.under = {(x, y, z) for x, y, z, p in self.cells if p in BLOCKS}   # (a block on one: laid after it)
        self.dug = [c for c in self.cells if c[3] == "dig"]
        self.dug_tiles = {c[:2] for c in self.dug}
        self.shells = {}                                      # (dz: the walls and doors standing at that height)
        for x, y, z, p in self.cells:
            if p in SHELL:
                self.shells.setdefault(z, []).append((x, y, z, p))

    @property
    def pieces(self) -> list:
        """(dx, dy, piece) of what is laid on the land and the roof over it (the hut's terms: rl/home.py, judge.laid)."""
        return [(x, y, p) for x, y, z, p in self.cells if z == 0 or (p == "roof" and z == STOREY)]

    @property
    def ground(self) -> list:
        """The stands on the land, (dx, dy): the first one is where the home is read from (rl/home.py inside)."""
        return [(x, y) for x, y, z in self.stands if z == 0]

    def box(self):
        """(x0, y0, x1, y1) of what must be open ground for it."""
        return min(x for x, _ in self.clear), min(y for _, y in self.clear), max(x for x, _ in self.clear), \
            max(y for _, y in self.clear)

    def reaches(self, s, cell) -> bool:
        """Is `cell` laid from stand `s`: in the game's reach, not a thing on the tile it stands on (a flight of
        stairs: from its foot only)."""
        dx, dy, dz, p = cell
        if cell in self.from_:
            return s in self.from_[cell]
        if s[:2] == (dx, dy) and (p not in BLOCKS or s[2] >= dz):
            return False
        return in_reach(dx - s[0], dy - s[1], dz - s[2], R)

    def usable(self, s, laid) -> bool:
        """Can one stand there now: the way to it laid (an upper storey's stairs, a dug tile's cells)."""
        return all(laid(c) for c in self.ways.get(s, ()))

    def stand_for(self, cell, open_=lambda s: True, laid=lambda c: True):
        """The stand nearest `cell` that reaches it (the game's reach, height counted): not on a thing's own tile, a
        spare one only when `open_`, one up a storey or down a dug way only once the way is laid; a flight of stairs or
        a dig from its own stand. None."""
        dx, dy, dz, p = cell
        if cell in self.from_:
            return next((s for s in self.from_[cell] if (open_(s) or s in self.stands) and self.usable(s, laid)), None)
        ok = [s for s in self.stands if self.usable(s, laid)] + [s for s in self.spare if open_(s)]
        ok = [s for s in ok if self.reaches(s, cell)]
        return min(ok, key=lambda s: (dx - s[0]) ** 2 + (dy - s[1]) ** 2 + ((dz - s[2]) / 2) ** 2, default=None)

    def held(self, cell, laid) -> bool:
        """A block over a storey goes down once that storey's walls and door all stand (a roof on an open ring is a
        roof on air: r61); a thing up a storey once the floor under it is laid; anything else, at once."""
        dx, dy, dz, p = cell
        if p == "seed":                                   # (sown in the soil tilled for it)
            return laid((dx, dy, dz, "till"))
        if p == "dig":
            return True                                   # (dug from its own stand, once that is dug: stand_for)
        if p not in BLOCKS:                               # (a thing up a storey stands on the floor laid under it)
            if dz < 0:                                    # (on a dug floor: once all is dug - a chair set early
                return all(laid(c) for c in self.dug)     # stands in the way of what is still to dig)
            return dz == 0 or p == "stairs" or laid((dx, dy, dz, "plank"))
        if (dx, dy, dz - 1) in self.under:                # (a roof's layer on the one under it: that one first)
            below = next(c for c in self.cells if c[:3] == (dx, dy, dz - 1))
            return laid(below)
        return all(laid(c) for c in self.shells.get(dz - STOREY, ()))


def of_pieces(pieces, stands, spare=()) -> Plan:
    """A plan on the land, from (dx, dy, piece): the hut's, a grid's. Its roof goes a storey over the land."""
    return Plan([(x, y, STOREY if p == "roof" else 0, p) for x, y, p in pieces], [(x, y, 0) for x, y in stands],
                [(x, y, 0) for x, y in spare])


def of_structure(s) -> Plan:
    """A build of storeys (rl/structure.py Structure)."""
    stands, spare = s.stands()
    if hasattr(s, "digs"):                                # (dug: rl/dug.py says its own ways)
        return Plan(s.cells(), stands, spare, clear=s.clear(), from_=s.from_(), ways=s.ways())
    feet = {(top[0], top[1], 0, "stairs"): tuple((x, y, 0) for x, y in at) for top, _, at, _ in s.stairs}
    up = {STOREY * k: (top[0], top[1], 0, "stairs") for top, _, _, k in s.stairs}
    return Plan(s.cells(), stands, spare, clear=s.clear(), from_=feet,
                ways={st: (up[st[2]],) for st in stands if st[2] in up})


def drawn(rows) -> Plan:
    """The plan of a grid the architect drew (rl/blueprint.py): its pieces, laid from the floor the door leads to once
    all is in (a tile the furniture closes round is never stood on: it would wall itself in) and from its path; spare,
    the ring round the walls outside."""
    from . import blueprint as bp
    at = lambda x, y: rows[y][x] if 0 <= y < len(rows) and 0 <= x < len(rows[y]) else " "      # noqa: E731
    near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]  # noqa: E731
    doors = [(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "D"]
    walked, stack = [], [q for d in doors for q in near(d) if at(*q) == "."]
    while stack:
        p = stack.pop()
        if p not in walked:
            walked.append(p)
            stack += [q for q in near(p) if at(*q) == "." and q not in walked]
    path = [(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "="]
    ps = bp.pieces(rows)
    shell = [(x, y) for x, y, p in ps if p in ("wall", "door", "fence")]
    x0, y0 = min(x for x, _ in shell) - 1, min(y for _, y in shell) - 1
    x1, y1 = max(x for x, _ in shell) + 1, max(y for _, y in shell) + 1
    spare = tuple((x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
                  if (x in (x0, x1) or y in (y0, y1)) and (x, y) not in path)
    return of_pieces(ps, tuple(sorted(walked, key=lambda q: (q[1], q[0]))) + tuple(path), spare)
