"""The build judge (#26, v1: does it work?): a build read from the world as facts (report), then as checks a player
would name, each passed or not, with the fact behind it. No weights: the grade is the checks passed. v2 (does it make
sense, #31) and v3 (does it look good, #32) add checks here, on the same report.

What closes a room, what gives light and how a room is shaped are the game's (Simulation.ENCLOSE, wild.LIGHT and
LIGHT_R, and sheltered's own size and compactness): read, not changed."""

from collections import Counter

from nexel_worlds.sim.core import Simulation
from nexel_worlds.sim.wild import LIGHT, LIGHT_R
from nexel_worlds.tiles import BLOCKS, DIRS, GROUNDS

ENCLOSE = Simulation.ENCLOSE
CAP, SIDE, COMPACT = 160, 14, 0.6           # (as the game's sheltered: past CAP tiles it is open land; a room's box)
_SOLID = {b.name: b.solid for b in BLOCKS}
_WALK = {g.name: g.walkable for g in GROUNDS}

ROOFED = 0.9                # of the inside under a roof
AREA = 4                    # tiles inside, at least: room to stand, sleep and turn round
CHECKS = ("closed", "door", "roofed", "lit", "bed", "room_enough")


def inside(world, x, y) -> set | None:
    """The tiles a flood fill from (x, y) reaches without crossing an ENCLOSE piece, or None past CAP."""
    seen, stack = {(x, y)}, [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if len(seen) > CAP:
            return None
        for dx, dy in DIRS.values():
            n = (cx + dx, cy + dy)
            if n not in seen and world.block(*n) not in ENCLOSE:
                seen.add(n)
                stack.append(n)
    return seen


def report(world, x, y) -> dict:
    """The build around (x, y), as facts: {"closed": True, "area", "box", "doors", "walls": {piece: n}, "roofed":
    share of the inside covered, "lights": sources within LIGHT_R of its middle, "contents": {block: n}}, or
    {"closed": False}."""
    tiles = inside(world, x, y)
    if tiles is None:
        return {"closed": False}
    xs, ys = [p[0] for p in tiles], [p[1] for p in tiles]
    w, h = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    if w > SIDE or h > SIDE or len(tiles) < COMPACT * w * h:
        return {"closed": False}
    edge = {n: world.block(*n) for cx, cy in tiles for dx, dy in DIRS.values() if (n := (cx + dx, cy + dy)) not in tiles}
    pieces = Counter(edge.values())
    mx, my = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
    lights = sum(world.block(mx + dx, my + dy) in LIGHT for dx in range(-LIGHT_R, LIGHT_R + 1)
                 for dy in range(-LIGHT_R, LIGHT_R + 1) if dx * dx + dy * dy <= LIGHT_R * LIGHT_R)
    return {"closed": True, "area": len(tiles), "box": (w, h),
            "doors": sum(n for k, n in pieces.items() if not _SOLID.get(k, True)),
            "walls": {k: n for k, n in pieces.items() if _SOLID.get(k, True)},
            "roofed": round(sum(1 for p in tiles if world.roof(*p)) / len(tiles), 3), "lights": lights,
            "contents": dict(Counter(b for b in (world.block(*p) for p in tiles) if b != "none"))}


def verdict(world, x, y) -> dict:
    """The build around (x, y): {"grade": checks passed, "of": how many, "checks": {name: bool}, "failed": [names],
    "report": the game's facts}."""
    r = report(world, x, y)
    c = r["closed"]
    checks = {"closed": c,
              "door": c and r["doors"] > 0,
              "roofed": c and r["roofed"] >= ROOFED,
              "lit": c and r["lights"] > 0,
              "bed": c and r["contents"].get("bed", 0) > 0,
              "room_enough": c and r["area"] >= AREA}
    return {"grade": sum(checks.values()), "of": len(CHECKS), "checks": checks,
            "failed": [k for k in CHECKS if not checks[k]], "report": r}


def told(v: dict) -> str:
    """The verdict in words, for the mind's facts: "closed, a door, no roof (40%), no light, no bed (grade 2 of 6)"."""
    r, ch = v["report"], v["checks"]
    if not r["closed"]:
        return "not closed"
    parts = ["closed", "a door" if ch["door"] else "no door",
             "roofed" if ch["roofed"] else f"no roof ({r['roofed']:.0%} covered)",
             "lit" if ch["lit"] else "no light", "a bed" if ch["bed"] else "no bed",
             f"{r['area']} tiles inside"]
    return ", ".join(parts) + f" (grade {v['grade']} of {v['of']})"


# ---- v2 (#31): does it make sense? A home's core things, laid out to be used, of a size for one, kept for its purpose.
# Its own checks beside v1's (verdict, which the reward reads: rl/core.py, Potential), on the same report.
STATIONS = ("workbench", "campfire")       # to make things at: in the home or by its door
SEATS = ("table", "chair")
CORE = ("bed", "chest", *LIGHT, *STATIONS, *SEATS)
FIT = (6, 40)               # tiles inside for one person: more than a box to stand in, less than a hall
NEAR = 3                    # tiles: a station by the door, a light by the bed
SENSE = ("chest", "station", "seat", "reachable", "door_out", "fit", "tidy", "light_by_bed")


def walkable(world, x, y) -> bool:
    """A tile you can stand on: ground you walk on (not water) and nothing solid on it."""
    return _WALK.get(world.ground(x, y), False) and not _SOLID.get(world.block(x, y), True)


def sense(world, x, y) -> dict:
    """The build around (x, y), v2: {"grade", "of", "checks": {name: bool}, "failed": [names], "facts"}. facts: the
    core things inside, counted ("core"), the doors, which core things no path from a door reaches ("unreached"),
    the doors that open onto no walkable ground ("shut_doors"), the tiles where land stands in the room over its floor
    ("stray": a block of earth or stone put down, landblocks.py: it raises the land)."""
    tiles = inside(world, x, y)
    r = report(world, x, y)
    if not r["closed"]:
        checks = dict.fromkeys(SENSE, False)
        return {"grade": 0, "of": len(SENSE), "checks": checks, "failed": list(SENSE), "facts": {"closed": False}}
    at = {p: world.block(*p) for p in tiles}
    doors = sorted(n for cx, cy in tiles for dx, dy in DIRS.values()
                   if (n := (cx + dx, cy + dy)) not in tiles and not _SOLID.get(world.block(*n), True))
    things = {p: b for p, b in at.items() if b in CORE}
    # on foot from a door: through the inside's walkable tiles; a thing is reached from a tile beside it
    walked, stack = set(), [n for d in doors for n in _beside(d) if n in tiles and walkable(world, *n)]
    while stack:
        p = stack.pop()
        if p not in walked:
            walked.add(p)
            stack += [n for n in _beside(p) if n in tiles and n not in walked and walkable(world, *n)]
    unreached = sorted({b for p, b in things.items() if not any(n in walked for n in _beside(p))})
    outside = lambda d: [n for n in _beside(d) if n not in tiles and n not in doors]           # noqa: E731
    shut = [d for d in doors if not any(walkable(world, *n) for n in outside(d))]
    near_door = {world.block(d[0] + i, d[1] + j) for d in doors for i in range(-NEAR, NEAR + 1) for j in range(-NEAR, NEAR + 1)}
    beds = [p for p, b in at.items() if b == "bed"]
    lit_bed = any(world.block(bx + i, by + j) in LIGHT for bx, by in beds for i in range(-NEAR, NEAR + 1) for j in range(-NEAR, NEAR + 1))
    core = dict(Counter(b for b in things.values()))
    floor = Counter(world.level(*p) for p in tiles).most_common(1)[0][0]
    stray = sorted(p for p in tiles if world.level(*p) > floor)
    checks = {"chest": core.get("chest", 0) > 0,
              "station": any(core.get(k) for k in STATIONS) or bool(near_door & set(STATIONS)),
              "seat": any(core.get(k) for k in SEATS),
              "reachable": bool(doors) and not unreached,          # (nothing in it: nothing out of reach; the lack is chest's, seat's)
              "door_out": bool(doors) and not shut,
              "fit": FIT[0] <= r["area"] <= FIT[1],
              "tidy": not stray,
              "light_by_bed": lit_bed}
    return {"grade": sum(checks.values()), "of": len(SENSE), "checks": checks, "failed": [k for k in SENSE if not checks[k]],
            "facts": {"closed": True, "area": r["area"], "core": core, "doors": doors, "unreached": unreached,
                      "shut_doors": shut, "stray": stray}}


def _beside(p):
    return [(p[0] + dx, p[1] + dy) for dx, dy in DIRS.values()]


def laid(world, home) -> int:
    """How many pieces of the hut's blueprint (rl/build.py PLAN) stand where it puts them, read from the world."""
    from .build import PLAN, WALLS
    hx, hy = home
    n = 0
    for dx, dy, piece in PLAN:
        x, y = hx + dx, hy + dy
        b = world.block(x, y)
        n += bool(world.roof(x, y)) if piece == "roof" else b in WALLS if piece == "wall" else \
            b in ENCLOSE and not _SOLID.get(b, True) if piece == "door" else b == piece
    return n


class Potential:
    """The build reward (#29): UNIT for each check the build passes, paid as the grade changes (a wall torn down pays
    back what it earned: potential-based, as rl/progress.py). Measured at one place - the home (rl/build.py), else the
    last closed build it stood in - so walking out of the room is no loss. Read after a place or a dig only."""

    UNIT = 0.25
    PIECE = 0.05                # and for each piece of the home's blueprint laid (rl/build.py): a ring of walls pays
    #                             as it rises, not only once it closes (r53 picked build 0.4% of the times it could)

    def __init__(self):
        self.at, self.phi = None, 0.0

    def step(self, world, home, me: tuple, results) -> float:
        if not any(t in ("place", "dig") and r.get("ok") for t, _, r in results):
            return 0.0
        at = (home[0] + 1, home[1] + 1) if home else self.at or me
        v = verdict(world, *at)
        if not home and not v["checks"]["closed"] and at != me:     # (the old spot is gone: where it stands now?)
            at, v = me, verdict(world, *me)
        if v["checks"]["closed"] and not home:
            self.at = at
        phi = self.UNIT * v["grade"] + (self.PIECE * laid(world, home) if home else 0.0)
        r, self.phi = phi - self.phi, phi
        return r
