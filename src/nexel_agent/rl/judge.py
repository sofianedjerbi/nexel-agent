"""The build judge (#26, v1: does it work?): a build read from the world as facts (report), then as checks a player
would name, each passed or not, with the fact behind it. No weights: the grade is the checks passed. v2 (does it make
sense, #31) and v3 (does it look good, #32) add checks here, on the same report.

What closes a room, what gives light and how a room is shaped are the game's (Simulation.ENCLOSE, wild.LIGHT and
LIGHT_R, and sheltered's own size and compactness): read, not changed."""

from collections import Counter

from nexel_worlds.sim.core import Simulation
from nexel_worlds.sim.wild import LIGHT, LIGHT_R
from nexel_worlds.tiles import BLOCKS, DIRS

ENCLOSE = Simulation.ENCLOSE
CAP, SIDE, COMPACT = 160, 14, 0.6           # (as the game's sheltered: past CAP tiles it is open land; a room's box)
_SOLID = {b.name: b.solid for b in BLOCKS}

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


class Potential:
    """The build reward (#29): UNIT for each check the build passes, paid as the grade changes (a wall torn down pays
    back what it earned: potential-based, as rl/progress.py). Measured at one place - the home (rl/build.py), else the
    last closed build it stood in - so walking out of the room is no loss. Read after a place or a dig only."""

    UNIT = 0.25

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
        phi = self.UNIT * v["grade"]
        r, self.phi = phi - self.phi, phi
        return r
