"""Where a build goes (rl/build.py): the nearest spot in view where all it stands on is open, flat land, away from
water (it runs onto flat land: nexel_worlds/flow.py); kept flat, a pit dug in it since found to be filled."""

import math

from nexel_worlds.sim.constants import VIEW_R
from nexel_worlds.sim.observe import WATER, YOU
from nexel_worlds.tiles import GROUNDS
from nexel_worlds.tiles import REACH as TILE_REACH

REACH = TILE_REACH["use"]          # (a piece is laid from where it stands: the hut's corners are this far from its middle)
WET = 8                     # tiles from water a build is not sited
LIQUID_CH = {g.char for g in GROUNDS if g.liquid}                                       # (no home on water, nor its path)


def _at(obs, key, X, Y):
    me, rows = obs["self"], obs.get(key)
    gx, gy = X - me["x"] + VIEW_R, Y - me["y"] + VIEW_R
    if not rows or not (0 <= gy < len(rows) and 0 <= gx < len(rows[gy])):
        return None
    return rows[gy][gx]


def flat(obs, X, Y, plan=None) -> bool:
    """The hut whose inside's top-left is (X, Y) stands on one level, clear of a hill's rim: no lower tile beside it,
    diagonals too (the game builds nothing there: world.rim). A pit dug in it since makes it no site."""
    plan = plan or _hut()
    from .actions import LEVEL
    if not obs.get("heights"):
        return True                                          # (no relief in view at all)
    sx, sy = plan.ground[0]
    lv = LEVEL.get(_at(obs, "heights", X + sx, Y + sy))
    if lv is None:
        return False
    for dx, dy in plan.clear | plan.rim:                     # (its own tiles and those round them: a compound's
        h = LEVEL.get(_at(obs, "heights", X + dx, Y + dy))   # ground between its parts is the land's own)
        if h is None or (dx, dy) in plan.dug_tiles:          # (its own trench and room dug: no pit to fill)
            continue                                         # (out of view: unknown, not lower - r59 left home after
        if h < lv or ((dx, dy) in plan.clear and h != lv):   # home each time it walked 9 tiles off)
            return False
    return True


def _pit(obs, X, Y, plan=None):
    """A tile in reach of a stand that is lower than the spot, on or beside it: ((x, y), the stand), or None."""
    plan = plan or _hut()
    from .actions import LEVEL
    sx, sy = plan.ground[0]
    lv = LEVEL.get(_at(obs, "heights", X + sx, Y + sy))
    for dx, dy in sorted(plan.clear | plan.rim, key=lambda q: (q[0], q[1])):
        h = LEVEL.get(_at(obs, "heights", X + dx, Y + dy))
        if (dx, dy) in plan.dug_tiles:
            continue
        s = next((s for s in plan.ground if max(abs(s[0] - dx), abs(s[1] - dy)) <= REACH), None)
        if h is not None and lv is not None and h < lv and s is not None:
            return (X + dx, Y + dy), s
    return None


def site(obs, ground_ch, plan=None) -> tuple | None:
    """The plan's origin (the hut's inside top-left) at the nearest spot in view where it all stands on flat, open
    ground."""
    plan = plan or _hut()
    me = obs["self"]
    best = None
    x0, y0, x1, y1 = plan.box()
    sx, sy = plan.ground[0]
    wet = _near_water(obs)
    for oy in range(-VIEW_R + 1 - y0, VIEW_R - 1 - y1):
        for ox in range(-VIEW_R + 1 - x0, VIEW_R - 1 - x1):
            X, Y = me["x"] + ox, me["y"] + oy
            if all(_at(obs, "grid", X + dx, Y + dy) in (ground_ch - LIQUID_CH) | {YOU} and (X + dx, Y + dy) not in wet
                   for dx, dy in plan.clear) and flat(obs, X, Y, plan):
                d = math.hypot(ox + sx, oy + sy)
                if best is None or d < best[0]:
                    best = (d, (X, Y))
    return best and best[1]


def _near_water(obs) -> set:
    """The tiles water in view may run onto: a river's, a lake's, running water's, and WET tiles round each (the game's
    water runs 7 tiles over flat land from its source: nexel_worlds/flow.py) - a build there floods (#69)."""
    me, rows = obs["self"], obs.get("grid") or []
    water = [(me["x"] + gx - VIEW_R, me["y"] + gy - VIEW_R) for gy, r in enumerate(rows) for gx, ch in enumerate(r)
             if ch in LIQUID_CH or ch == WATER]
    return {(x + i, y + j) for x, y in water for i in range(-WET, WET + 1) for j in range(-WET, WET + 1)}


def _hut():
    from .build import HUT
    return HUT
