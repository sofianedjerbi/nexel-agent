"""Where a build goes (rl/build.py): the nearest spot in view where all it stands on is open, flat land, away from
water (it runs onto flat land: nexel_worlds/flow.py); kept flat, a pit dug in it since found to be filled."""

import math

from nexel_worlds.sim.constants import VIEW_R
from nexel_worlds.sim.storeys import STOREY
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
    if not obs.get("heights"):
        return True                                          # (no relief in view at all)
    lv = next((h for sx, sy in (*plan.ground, *sorted(plan.clear)) if (h := _land(obs, X, Y, plan, sx, sy))
               is not None), None)                 # (its ground out of view: read on what of it is in view - the T on
    if lv is None:                                 # seed 203 gave up four sites gathering beside them; none in view:
        return True                                # unknown, not uneven)
    for dx, dy in plan.clear | plan.rim:                     # (its own tiles and those round them: a compound's
        h = _land(obs, X, Y, plan, dx, dy)                   # ground between its parts is the land's own)
        if h is None or (dx, dy) in plan.dug_tiles:          # (its own trench and room dug: no pit to fill)
            continue                                         # (out of view: unknown, not lower - r59 left home after
        if h < lv or ((dx, dy) in plan.clear and h != lv):   # home each time it walked 9 tiles off)
            return False
        if h - lv > _over(plan).get((dx, dy), 99) - STOREY:  # (a block over the land leaves a storey to stand under
            return False                                     # it, the game's rule: a hill under the eaves, unroofed)
    return True


def _over(plan) -> dict:
    """{(dx, dy): the lowest layer of a block over that tile} beyond the plan's own ground: its roof's overhang."""
    if getattr(plan, "_over", None) is None:
        plan._over = {}
        for x, y, z, _ in plan.cells:
            if z > 0 and (x, y) not in plan.clear:
                plan._over[(x, y)] = min(z, plan._over.get((x, y), z))
    return plan._over


def level_at(obs, X, Y) -> int:
    """The land's level at a tile, from the observation's heights (none sent: all the low ground, 0)."""
    from .actions import LEVEL
    return LEVEL.get(_at(obs, "heights", X, Y), 0) if obs.get("heights") else 0


def _built_over(plan) -> dict:
    """{(dx, dy): the lowest layer of the plan's own land laid over that tile (stone, earth: what the heights show)}."""
    from .structure import LAND_IN_CELL
    if getattr(plan, "_built_over", None) is None:
        plan._built_over = {}
        for x, y, z, p in plan.cells:
            if z > 0 and p in LAND_IN_CELL:
                plan._built_over[(x, y)] = min(z, plan._built_over.get((x, y), z))
    return plan._built_over


def _land(obs, X, Y, plan, dx, dy):
    """The land's level at the plan's tile (dx, dy) as the view's heights show it, or None: out of view, or read on
    the build itself - a stone or earth roof is land laid up there, and the heights show its top (live trial 10: from
    the far end of a stone-roofed house the land under the roof read 7 high, the site 'dug into', 45 minutes of walks
    to fill a pit that was the roof)."""
    from .actions import LEVEL
    h = LEVEL.get(_at(obs, "heights", X + dx, Y + dy))
    return None if h is not None and h >= _built_over(plan).get((dx, dy), 99) else h


def _pit(obs, X, Y, plan=None):
    """A tile in reach of a stand that is lower than the spot, on or beside it: ((x, y), the stand), or None."""
    plan = plan or _hut()
    sx, sy = plan.ground[0]
    lv = _land(obs, X, Y, plan, sx, sy)
    for dx, dy in sorted(plan.clear | plan.rim, key=lambda q: (q[0], q[1])):
        h = _land(obs, X, Y, plan, dx, dy)
        if (dx, dy) in plan.dug_tiles:
            continue
        s = next((s for s in plan.ground if max(abs(s[0] - dx), abs(s[1] - dy)) <= REACH), None)
        if h is not None and lv is not None and h < lv and s is not None:
            return (X + dx, Y + dy), s
    return None


def climb_out(obs, social, tx, ty, dirs, tool_for, ground_ch):
    """Every walk toward (tx, ty) found no way (a hollow deeper than a climb: the game drops any height, climbs one
    layer - seed 704, a round home, fell 3 into one walking home): a tile on toward it, the side the nearest its way
    first - stepped onto if a climb, else dug down a layer (as a player cuts steps out of a pit), or None."""
    from .actions import LEVEL
    me = obs["self"]
    here = LEVEL.get(_at(obs, "heights", me["x"], me["y"]))
    if here is None:
        return None
    for d, (i, j) in sorted(dirs.items(), key=lambda kv: -(kv[1][0] * tx + kv[1][1] * ty)):
        X, Y = me["x"] + i, me["y"] + j
        h = LEVEL.get(_at(obs, "heights", X, Y))
        if h is None or i * tx + j * ty <= 0 or _at(obs, "grid", X, Y) not in ground_ch:
            continue                                         # (on toward it, bare ground)
        if here <= h <= here + 1:
            no_way(social).clear()                           # (a step up: the ways found shut from below, tried again)
            return "move_to", {"x": X, "y": Y}
        if h > here + 1:
            return "dig", {"dir": d, "z": h, "x": X, "y": Y, **({"with": w} if (w := tool_for(obs, d)) else {})}
    return None


def _bump(obs, X, Y, plan, ground_ch):
    """A tile of the site higher than its land, with a lower one of it beside to dig it down from: ((x, y, its
    level), (x, y) the tile beside, the way from there to it), or None (the inverse of _pit)."""
    lv = next((h for p in (*plan.ground, *sorted(plan.clear)) if (h := _land(obs, X, Y, plan, *p)) is not None), None)
    if lv is None:
        return None
    ways = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
    for dx, dy in sorted(plan.clear):
        h = _land(obs, X, Y, plan, dx, dy)
        if h is None or h <= lv or (dx, dy) in plan.dug_tiles or (dx, dy) in getattr(plan, "steps", ()) \
                or _at(obs, "grid", X + dx, Y + dy) not in ground_ch:
            continue                                         # (bare land: not a flight of its stairs, nothing on it)
        for d, (i, j) in ways.items():                       # (from a tile at the land's level, open, in view)
            sx, sy = X + dx - i, Y + dy - j
            if _land(obs, X, Y, plan, dx - i, dy - j) == lv and _at(obs, "grid", sx, sy) in ground_ch | {YOU}:
                return (X + dx, Y + dy, h), (sx, sy), d
    return None


def site(obs, ground_ch, plan=None, clearable=frozenset()) -> tuple | None:
    """The plan's origin (the hut's inside top-left) at the nearest spot in view where it stands on flat, open ground.
    A build wider than the view (an annex beside a house: 20 tiles, the view 19) is judged on what of it is in view,
    its first floor tile in it: the rest is seen walking there, and mended or the spot given up (rl/build.py) - it
    stood on 5 of 10 fresh seeds only where the hut's spot, chosen first, happened to hold it."""
    plan = plan or _hut()
    me = obs["self"]
    best = None
    x0, y0, x1, y1 = plan.box()
    sx, sy = plan.ground[0]
    wide = x1 - x0 > 2 * VIEW_R - 3 or y1 - y0 > 2 * VIEW_R - 3
    wet = _near_water(obs)
    for oy in range(-VIEW_R + 1 - sy, VIEW_R - sy) if wide else range(-VIEW_R + 1 - y0, VIEW_R - 1 - y1):
        for ox in range(-VIEW_R + 1 - sx, VIEW_R - sx) if wide else range(-VIEW_R + 1 - x0, VIEW_R - 1 - x1):
            X, Y = me["x"] + ox, me["y"] + oy
            chs = [_at(obs, "grid", X + dx, Y + dy) for dx, dy in plan.clear]
            if all(ch in (ground_ch - LIQUID_CH) | {YOU} | clearable or wide and ch is None for ch in chs) \
                    and not any((X + dx, Y + dy) in wet for dx, dy in plan.clear) and flat(obs, X, Y, plan):
                d = math.hypot(ox + sx, oy + sy) + 2 * sum(ch in clearable for ch in chs) + sum(ch is None for ch in chs)
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


TRIES = 4                     # (a dig refused this often from one stand: tried from another side - seed 21's mushroom,
                              # "behind the land", dug at 199 times from the same tile; a dig still under way is no try:
                              # a rock takes 54 ticks, the T's on seed 7 was given up and walked from forever)


def tried(social, X, Y, here, count=True, at=None) -> bool:
    """Has clearing (X, Y) from `here` been tried TRIES times (counting this try, unless count is False)? A try is
    counted once at the game's tick `at`: the home's step is asked for more than once a decision (the mask, the act,
    the step after a way found shut), and a stump dug once from a doorway was given up as dug at four times."""
    t = social.__dict__.setdefault("clear_tries", {})
    last = social.__dict__.setdefault("clear_at", {})
    if t.get((X, Y, here), 0) >= TRIES:
        return True
    if count and (at is None or last.get((X, Y, here)) != at):
        t[(X, Y, here)] = t.get((X, Y, here), 0) + 1
        last[(X, Y, here)] = at
    return False


def no_way(social) -> set:
    """Tiles a walk of the builder's found no way to (a workbench walled in): crafting looks past them."""
    return social.__dict__.setdefault("no_way", set()) if social is not None else set()


def fresh(social, laid: int):
    """Ways found shut are tried again once the build has grown (`laid` pieces): a stand up a storey has no way to it
    until its flight is laid - the tower on seed 304, its last roof cell's stand shut for good, one cell left."""
    if social.__dict__.get("laid_n") != laid:
        social.laid_n = laid
        no_way(social).clear()
        social.__dict__.get("clear_tries", {}).clear()


def walked_nowhere(social, to: dict, me: dict) -> bool:
    """A walk ordered TRIES times from the same tile: the way is shut (the T on seed 7: a walk to a workbench with no
    path, ordered 199 times); kept in no_way."""
    if tried(social, to["x"], to["y"], (me["x"], me["y"])):
        no_way(social).add((to["x"], to["y"]))
        return True
    return False


def way_out(obs, ground_ch, keep, dirs, tool_for, clearable=frozenset()):
    """Walled in - its own wall, a rock, a bush on every side (the T on seed 7: every walk "no path", 199 times) - a
    dig through what is not the home's, the nearest way out; None when a side is open."""
    me, grid = obs["self"], obs.get("grid") or []
    if not grid or any(_at(obs, "grid", me["x"] + dx, me["y"] + dy) in ground_ch for dx, dy in dirs.values()):
        return None
    d = next((k for k, (dx, dy) in dirs.items() if (me["x"] + dx, me["y"] + dy) not in keep), None) or next(
        (k for k, (dx, dy) in dirs.items() if _at(obs, "grid", me["x"] + dx, me["y"] + dy) in clearable), None)  # (a tree
    #  on its own ground, to be cleared anyway - the round home on seed 203: its walls and two trees round it)
    return ("dig", {"dir": d, **({"with": w} if (w := tool_for(obs, d)) else {})}) if d else None


SEARCH = tuple((dx * k, dy * k) for k in (1, 2)                # (a view's walk each way in turn, then two views)
               for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)))


def _look_further(obs, social, plan):
    """No spot in view for a drawn build: a view's walk from where it began looking, a way at a time, a look again
    there, as a player goes looking for flat land; the next way once there with none, or stuck (the sea in the way).
    The hut, the body's own, is sited in view or not at all, as it was trained."""
    if plan is _hut():
        return None
    me, t = obs["self"], obs.get("tick")
    pos = (me["x"], me["y"])
    s = getattr(social, "search", None)
    if s is None:
        s = social.search = {"from": pos, "i": 0, "seen": None, "stuck": 0}
    while s["i"] < len(SEARCH):
        dx, dy = SEARCH[s["i"]]
        tx, ty = s["from"][0] + dx * (VIEW_R - 1), s["from"][1] + dy * (VIEW_R - 1)
        if s["seen"] is not None and s["seen"][1] != t:      # (ticks went by: moved, or not)
            s["stuck"] = s["stuck"] + 1 if s["seen"][0] == pos else 0
        if s["seen"] is None or s["seen"][1] != t:
            s["seen"] = (pos, t)
        if max(abs(tx - pos[0]), abs(ty - pos[1])) <= 2 or s["stuck"] >= 3:
            s["i"], s["stuck"], s["seen"] = s["i"] + 1, 0, None
            continue
        return "move_to", {"x": tx, "y": ty}
    if max(abs(pos[0] - s["from"][0]), abs(pos[1] - s["from"][1])) > VIEW_R:   # (none round there: it goes on looking
        social.search = None                                 # from where it got to - seed 702, a two-storey house)
        return _look_further(obs, social, plan)
    return None
