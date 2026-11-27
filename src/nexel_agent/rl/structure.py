"""Builds in three dimensions (#66): a design (the architect's choices) as storeys of tiles over one footprint, a
rectangle or a round, drawn by code and laid as the game builds up (nexel_worlds/sim/storeys.py, climbing.py):

    storey 0      walls, a door, furniture on the land
    storey k > 0  a floor of planks laid on the walls under it (STOREY layers up), walls, a door, furniture on that,
                  reached by a flight of stairs from the ground outside, to its door (2 to 7 layers up: three storeys)
    roof          over the top storey and one tile past its walls (none: an open yard, a base): flat, or a gable
                  stepping up a layer a row to a ridge along its long side, or a hip stepping up on all four (a dome
                  on a round); shingles, planks, stone or earth blocks, the eaves and the ridge in another if chosen

A cell is (dx, dy, dz, piece): dz the layers over the home's land, a thing standing at that height (on the land, or
on a floor block there) or a block in that cell. Blocks go down in the order the game holds them: over the walls
first (a block rests on a wall a storey under it), then each beside one laid. Judged storey by storey with the
judge's own rules (rl/judge.py), on a view of each storey of a sim world it is stamped in (StoreyView)."""

import math

from nexel_worlds.sim.storeys import STOREY, UPSTAIRS
from nexel_worlds.tiles import PLACEABLE

from . import blueprint as bp

SHAPES = ("rect", "round")
MOST_STOREYS = 3            # the top floor 6 layers up: a flight of stairs leads 7 at most (climbing.RISE)
SIDE = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
THINGS = ("wall", "door", "fence", *bp.FURNITURE)
BLOCKS = ("plank", "roof")
ROOF_STYLES = ("flat", "gable", "hip")
ROOF_MATERIALS = ("roof", "plank", "stone_block", "dirt")     # (shingles, planks: blocks; stone, earth: land in a cell)
LAND_IN_CELL = ("stone_block", "dirt")
ROOF_RISE = 3               # layers a roof steps up at the most: its top in reach from the floor under it
UPSTAIRS_FURNITURE = tuple(f for f in bp.FURNITURE if PLACEABLE.get(f, f) in UPSTAIRS)


def footprint(spec) -> tuple:
    """(inside, ring): the inside tiles and the wall tiles round them, from the walls' top-left (0, 0)."""
    if spec.get("shape", "rect") == "round":
        if "radius" not in spec:
            raise ValueError("a round needs its radius")
        r = float(spec["radius"])
        if not 2 <= r <= 6:
            raise ValueError("radius from 2 to 6")
        n = math.ceil(r) + 1
        inside = {(x + n, y + n) for x in range(-n, n + 1) for y in range(-n, n + 1) if math.hypot(x, y) < r - .5}
    else:
        if "width" not in spec or "depth" not in spec:
            raise ValueError("a design needs its width and depth (or shape round and a radius)")
        w, d = int(spec["width"]), int(spec["depth"])
        if not (2 <= w <= bp.MOST - 2 and 2 <= d <= bp.MOST - 2):
            raise ValueError(f"width and depth from 2 to {bp.MOST - 2}")
        inside = {(x + 1, y + 1) for x in range(w) for y in range(d)}
    near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]   # noqa: E731
    ring = {q for p in inside for q in near(p) if q not in inside}
    ring |= {(x, y) for x, y in _around(inside) if (x, y) not in inside and sum(q in ring for q in near((x, y))) >= 2}
    return inside, ring          # (a round's corners filled where two walls meet across a diagonal: it looks whole)


def _around(ts):
    return {(x + i, y + j) for x, y in ts for i in (-1, 0, 1) for j in (-1, 0, 1)}


def door_at(inside, ring, side) -> tuple:
    """The wall tile in the middle of that side: the furthest that way, nearest the middle across."""
    dx, dy = SIDE[side]
    cx = (min(x for x, _ in ring) + max(x for x, _ in ring)) / 2
    cy = (min(y for _, y in ring) + max(y for _, y in ring)) / 2
    beside = [p for p in ring if (p[0] + dx, p[1] + dy) not in inside and (p[0] - dx, p[1] - dy) in inside]
    return max(beside, key=lambda p: (p[0] - cx) * dx + (p[1] - cy) * dy - abs((p[1] - cy) * dx + (p[0] - cx) * dy) * 2)


class Structure:
    """A design drawn: per storey its things {(x, y): piece}, its door, the floor and roof blocks, the stairs."""

    def __init__(self, spec: dict):
        self.spec, self.shape = spec, spec.get("shape", "rect")
        if self.shape not in SHAPES:
            raise ValueError(f"shape is one of {', '.join(SHAPES)}")
        self.n = int(spec.get("storeys", 1))
        if not 1 <= self.n <= MOST_STOREYS:
            raise ValueError(f"storeys from 1 to {MOST_STOREYS}")
        self.roof = roof_of(spec.get("roof", True))
        self.roofed = self.roof is not None
        self.inside, self.ring = footprint(spec)
        door = str(spec.get("door", "south")).lower()
        if door not in SIDE:
            raise ValueError(f"door is one of {', '.join(SIDE)}")
        sides = [door] + [s for s in ("east", "west", "north", "south") if s != door]
        self.doors = [door_at(self.inside, self.ring, sides[k]) for k in range(self.n)]
        self.sides = sides[:self.n]
        self.storeys = [self._storey(k) for k in range(self.n)]
        self.path = self._line(self.doors[0], SIDE[door], max(0, min(int(spec.get("path", 2)), 3)))
        self.stairs = []          # (the tile beside the door, the steps, where they are laid from: beside the second
        for k in range(1, self.n):                       # step, in reach of the first; the game runs them out, k)
            d = SIDE[self.sides[k]]
            steps = self._line(self.doors[k], d, STOREY * k - 1)
            at = steps[min(1, len(steps) - 1)]
            self.stairs.append((steps[0], steps, ((at[0] + d[1], at[1] + d[0]), (at[0] - d[1], at[1] - d[0])), k))

    @staticmethod
    def _line(p, d, n) -> list:
        return [(p[0] + d[0] * i, p[1] + d[1] * i) for i in range(1, n + 1)]

    def _storey(self, k) -> dict:
        """Storey k's things: walls, its door, the furniture chosen for it set where all is reached (bp.tidied)."""
        things = {p: "wall" for p in self.ring}
        things[self.doors[k]] = "door"
        mine = {n: [a[:2] for a in (ats if isinstance(ats, list) and ats and isinstance(ats[0], list) else [ats])
                    if (a[2] if len(a) > 2 else 0) == k]
                for n, ats in (self.spec.get("furniture") or {}).items()}
        for name, at in _tidied(self.inside, self.doors[k], {n: v for n, v in mine.items() if v}).items():
            if name not in bp.FURNITURE:
                raise ValueError(f"no such furniture: {name} (one of {', '.join(bp.FURNITURE)})")
            if k and PLACEABLE.get(name, name) not in UPSTAIRS:
                raise ValueError(f"a {name} needs the land under it: on storey {k}, only {', '.join(UPSTAIRS_FURNITURE)}")
            for x, y in at:
                things[(x, y)] = name
        return things

    def cells(self) -> list:
        """(dx, dy, dz, piece) from the walls' top-left, in the order laid (see the module's table)."""
        out = [(x, y, 0, p) for (x, y), p in _shell_first(self.storeys[0])] + [(x, y, 0, "path") for x, y in self.path]
        foot = self.inside | self.ring
        for k in range(1, self.n):
            under = {p for p, t in self.storeys[k - 1].items() if t in ("wall", "door")}
            out += [(x, y, STOREY * k, "plank") for x, y in _held(foot, under)]
            top, _, _, _ = self.stairs[k - 1]
            out += [(top[0], top[1], 0, "stairs")]
            out += [(x, y, STOREY * k, p) for (x, y), p in _shell_first(self.storeys[k])]
        if self.roofed:
            under = {p for p, t in self.storeys[-1].items() if t in ("wall", "door")}
            out += [(x, y, STOREY * self.n + i, m) for i, layer in enumerate(self.layers(_held(_around(foot), under)))
                    for x, y, m in layer]
        return out

    def layers(self, base) -> list:
        """The roof's layers, the lowest first: [[(x, y, material)]]. Flat, one; a gable, each a row in from the long
        sides; a hip, a ring in from every side; ROOF_RISE layers at the most (what is left at the top, flat)."""
        style, mat, eaves, ridge = self.roof["style"], self.roof["material"], self.roof.get("eaves"), self.roof.get("ridge")
        edge = {p for p, k in _rings(set(base)).items() if k == 0}   # (the eaves: the roof's outer ring, whatever its style)
        xs, ys = [x for x, _ in base], [y for _, y in base]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        along_x = max(xs) - min(xs) >= max(ys) - min(ys)           # (the ridge along the long side)
        if style == "flat":
            depth = lambda p: 0                                            # noqa: E731
        elif style == "gable":
            depth = lambda p: int((max(ys) - min(ys)) / 2 - abs(p[1] - cy)) if along_x else int((max(xs) - min(xs)) / 2 - abs(p[0] - cx))   # noqa: E731
        else:
            ring = _rings(set(base))
            depth = ring.get                                               # noqa: E731
        deck = mat in LAND_IN_CELL                                # (stone, earth: held by blocks only, not by a wall
        most = max(depth(p) for p in base)                       # standing under: laid on a deck of planks)
        top = min(ROOF_RISE - deck, most)                        # (a slope spread to the ridge: no flat top)
        rise = {p: round(depth(p) * top / most) if most else 0 for p in base}
        out = [[] for _ in range(top + 1 + deck)]
        for p in base:
            d = rise[p]
            if deck:
                out[0].append((p[0], p[1], eaves if eaves and p in edge else "plank"))
            for i in range(d + 1):
                m = eaves if eaves and i == 0 and p in edge else ridge if ridge and i == d and depth(p) == most and most else mat
                if not (deck and eaves and p in edge):           # land's next layer, the thing in its way: nexel-worlds#308)
                    out[i + deck].append((p[0], p[1], m))
        return [layer for layer in out if layer]

    def cost(self) -> int:
        """The walls and blocks it takes (floors, roof): what is gathered and made most."""
        return sum(p == "wall" or p in ROOF_MATERIALS for *_, p in self.cells())

    def stands(self) -> tuple:
        """(stands, spare): the floor each storey's door leads to (dx, dy, dz: the height one stands at) and the path;
        spare, the ground round the walls and where each flight of stairs is laid from."""
        st = []
        for k, things in enumerate(self.storeys):
            st += [(x, y, STOREY * k) for x, y in _walked(self.inside, self.doors[k], things)]
        st += [(x, y, 0) for x, y in self.path]
        flights = {p for _, steps, _, _ in self.stairs for p in steps}
        spare = [(x, y, 0) for x, y in sorted(_around(self.inside | self.ring) - self.inside - self.ring)
                 if (x, y) not in flights and (x, y) not in self.path]
        spare += [(x, y, 0) for _, _, feet, _ in self.stairs for x, y in feet]
        return tuple(st), tuple(spare)

    def from_(self) -> dict:
        """A flight of stairs, laid from beside its second step (the game runs it out from the floor it leads to)."""
        return {(top[0], top[1], 0, "stairs"): tuple((x, y, 0) for x, y in at) for top, _, at, _ in self.stairs}

    def ways(self) -> dict:
        """A stand up a storey: once that storey's stairs are laid."""
        up = {STOREY * k: (top[0], top[1], 0, "stairs") for top, _, _, k in self.stairs}
        return {st: (up[st[2]],) for st in self.stands()[0] if st[2] in up}

    def clear(self) -> set:
        """The tiles that must be open ground for it: what stands on the land, the path, the stairs and their foot."""
        return self.inside | self.ring | set(self.path) | {p for _, steps, _, _ in self.stairs for p in steps}

    def grid(self, k=0) -> list:
        """Storey k as the architect's grid (the floor '.', the legend's characters), for a look."""
        ch = {v: c for c, v in bp.LEGEND.items()}
        ts = self.inside | self.ring | set(self.path)
        x1, y1 = max(x for x, _ in ts), max(y for _, y in ts)
        rows = []
        for y in range(y1 + 1):
            rows.append("".join(ch.get(self.storeys[k].get((x, y)), "." if (x, y) in self.inside else "=" if
                                       k == 0 and (x, y) in self.path else " ") for x in range(x1 + 1)).rstrip())
        return rows


def roof_of(v):
    """The roof chosen: None (open), or {"style", "material", "eaves", "ridge"}; True is a flat roof of shingles."""
    if v is False or v is None or v == "none":
        return None
    r = {"style": "flat", "material": "roof"} if v is True else {"style": v, "material": "roof"} if isinstance(v, str) else dict(v)
    r.setdefault("style", "flat")
    r.setdefault("material", "roof")
    if r["style"] not in ROOF_STYLES:
        raise ValueError(f"a roof's style is one of {', '.join(ROOF_STYLES)}")
    for k in ("material", "eaves", "ridge"):
        if r.get(k) is not None and r[k] not in ROOF_MATERIALS:
            raise ValueError(f"a roof is of {', '.join(ROOF_MATERIALS)} (its {k}: {r[k]})")
    return r


def _rings(ts) -> dict:
    """How far in each tile is from the edge of `ts`, in rings (the edge 0)."""
    near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]   # noqa: E731
    out, left, d = {}, set(ts), 0
    while left:
        edge = {p for p in left if any(q not in left for q in near(p))}
        for p in edge:
            out[p] = d
        left -= edge
        d += 1
    return out


def _shell_first(things) -> list:
    return sorted(things.items(), key=lambda kv: (kv[1] not in ("wall", "door"), kv[0][1], kv[0][0]))


def _held(cells, under) -> list:
    """The blocks over `cells` in an order the game holds each: over a wall first, then each beside one laid."""
    near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]   # noqa: E731
    order, done = [], set()
    frontier = sorted(c for c in cells if c in under)
    while frontier:
        nxt = []
        for c in frontier:
            if c not in done:
                done.add(c)
                order.append(c)
                nxt += [q for q in near(c) if q in cells and q not in done]
        frontier = sorted(set(nxt))
    return order


def _walked(inside, door, things) -> list:
    near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]   # noqa: E731
    walked, stack = [], [q for q in near(door) if q in inside and q not in things]
    while stack:
        p = stack.pop()
        if p not in walked:
            walked.append(p)
            stack += [q for q in near(p) if q in inside and q not in things and q not in walked]
    return sorted(walked, key=lambda q: (q[1], q[0]))


def _tidied(inside, door, chosen) -> dict:
    """bp.tidied's rule on any footprint: each piece by a wall, where chosen or nearest, all reached from the door."""
    near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]   # noqa: E731
    front = next(q for q in near(door) if q in inside)
    by_wall = lambda q: 0 if any(n not in inside for n in near(q)) else 1                      # noqa: E731

    def reached(taken):
        walked, stack = set(), [front]
        while stack:
            p = stack.pop()
            if p not in walked:
                walked.add(p)
                stack += [q for q in near(p) if q in inside and q not in taken and q not in walked]
        return front not in taken and all(any(q in walked for q in near(t)) for t in taken)
    taken, out = set(), {}
    for name, ats in chosen.items():
        for at in ats:
            try:
                x, y = int(at[0]) + 1, int(at[1]) + 1                 # (chosen from the inside's top-left)
            except (TypeError, ValueError, IndexError):
                x, y = 1, 1
            ok = [q for q in sorted(inside) if q not in taken and reached(taken | {q})]
            if ok:
                q = min(ok, key=lambda q: (by_wall(q), abs(q[0] - x) + abs(q[1] - y)))
                taken.add(q)
                out.setdefault(name, []).append(q)
    return out
