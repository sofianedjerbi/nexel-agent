"""The `build` verb (#28): a blueprint's next step, as one game tool call - the "how" of a home, never the "when" (the
policy chooses build or anything else, and when to come back to it). A hut: a 3 x 3 inside, walled all round with a
door on the south side, a fire and a bed in two corners, a roof over all of it, laid the way a player does it,
standing in the middle where every piece is in reach (tiles.REACH "use", the lead counted).

Read from the game: what closes a room (Simulation.ENCLOSE, a wall the solid ones, a door the others), what each
piece is made of (tiles.RECIPES, through pursue: a piece the pack lacks is made, or what it takes is gathered), what
is laid already (the observation's grid, and its roofs). Its one memory is where the home is: social.home, chosen once,
on flat open ground near where it stands. None: no step to take now (nowhere flat in view, nothing it can lay or make).

The hut is the default plan; a home the architect drew (rl/blueprint.py, #66) is another, kept in social.plan: its
pieces from the grid's top-left and the floor tiles it is laid from, each piece from the nearest one in reach."""

import math

from nexel_worlds.sim.constants import VIEW_R
from nexel_worlds.sim.observe import ITEMS, YOU
from nexel_worlds.sim.storeys import EARTH, STOREY
from nexel_worlds.tiles import GROUNDS
from nexel_worlds.tiles import REACH as TILE_REACH
from nexel_worlds.tiles import BLOCKS

from .judge import ENCLOSE
from .plan import BLOCKS as CELL_BLOCKS
from .plan import Plan, of_pieces
from .structure import LAND_IN_CELL
from .pursue import craft

W = H = 3                                                   # the inside
DOOR = (1, H)                                               # in the south wall's middle
FURNITURE = ((0, 0, "campfire"), (W - 1, 0, "bed"))         # a corner each
FAR = 64                    # a home further off is given up (fallen, back at spawn; an errand for wood goes 30+)
_SOLID = {b.name: b.solid for b in BLOCKS}
CH = {b.name: b.char for b in BLOCKS}
WALLS = tuple(k for k in ENCLOSE if _SOLID[k])
DOORS = tuple(k for k in ENCLOSE if not _SOLID[k])
WALL_CH, DOOR_CH = {CH[k] for k in WALLS}, {CH[k] for k in DOORS}
LIQUID_CH = {g.char for g in GROUNDS if g.liquid}                                       # (no home on water, nor its path)
PATH_CH = next(g.char for g in GROUNDS if g.name == "cobble")                           # (what the path item lays)
BLOCK_CH = {b.char for b in BLOCKS if b.name != "none" and b.hardness > 0}   # (what stands in the way and can be dug: a
#                                                                             tree, a stump, a flower grown on the spot)


def blueprint() -> list:
    """(dx, dy, piece) from the inside's top-left, in the order it is laid: walls, door, furniture, then the roof."""
    ring = [(dx, dy) for dy in range(-1, H + 1) for dx in range(-1, W + 1) if dx in (-1, W) or dy in (-1, H)]
    return ([(dx, dy, "wall") for dx, dy in ring if (dx, dy) != DOOR] + [(*DOOR, "door")]
            + [(dx, dy, p) for dx, dy, p in FURNITURE]
            + [(dx, dy, "roof") for dy in range(-1, H + 1) for dx in range(-1, W + 1)])


PLAN = blueprint()
REACH = TILE_REACH["use"]          # (a piece is laid from where it stands: the hut's corners are this far from its middle)


HUT = of_pieces(PLAN, ((W // 2, H // 2),))


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


def plan_of(social) -> Plan:
    return getattr(social, "plan", None) or HUT


def _at(obs, key, X, Y):
    me, rows = obs["self"], obs.get(key)
    gx, gy = X - me["x"] + VIEW_R, Y - me["y"] + VIEW_R
    if not rows or not (0 <= gy < len(rows) and 0 <= gx < len(rows[gy])):
        return None
    return rows[gy][gx]


def flat(obs, X, Y, plan=HUT) -> bool:
    """The hut whose inside's top-left is (X, Y) stands on one level, clear of a hill's rim: no lower tile beside it,
    diagonals too (the game builds nothing there: world.rim). A pit dug in it since makes it no site."""
    from .actions import LEVEL
    if not obs.get("heights"):
        return True                                          # (no relief in view at all)
    sx, sy = plan.ground[0]
    lv = LEVEL.get(_at(obs, "heights", X + sx, Y + sy))
    if lv is None:
        return False
    x0, y0, x1, y1 = plan.box()
    for dx in range(x0 - 1, x1 + 2):
        for dy in range(y0 - 1, y1 + 2):
            h = LEVEL.get(_at(obs, "heights", X + dx, Y + dy))
            if h is None:
                continue                                     # (out of view: unknown, not lower - r59 left home after
            if h < lv or (x0 <= dx <= x1 and y0 <= dy <= y1 and h != lv):   # home each time it walked 9 tiles off)
                return False
    return True


def _pit(obs, X, Y, plan=HUT):
    """A tile in reach of a stand that is lower than the spot, on or beside it: ((x, y), the stand), or None."""
    from .actions import LEVEL
    sx, sy = plan.ground[0]
    lv = LEVEL.get(_at(obs, "heights", X + sx, Y + sy))
    x0, y0, x1, y1 = plan.box()
    for dx in range(x0 - 1, x1 + 2):
        for dy in range(y0 - 1, y1 + 2):
            h = LEVEL.get(_at(obs, "heights", X + dx, Y + dy))
            s = next((s for s in plan.ground if max(abs(s[0] - dx), abs(s[1] - dy)) <= REACH), None)
            if h is not None and lv is not None and h < lv and s is not None:
                return (X + dx, Y + dy), s
    return None


def site(obs, ground_ch, plan=HUT) -> tuple | None:
    """The plan's origin (the hut's inside top-left) at the nearest spot in view where it all stands on flat, open
    ground."""
    me = obs["self"]
    best = None
    x0, y0, x1, y1 = plan.box()
    sx, sy = plan.ground[0]
    for oy in range(-VIEW_R + 1 - y0, VIEW_R - 1 - y1):
        for ox in range(-VIEW_R + 1 - x0, VIEW_R - 1 - x1):
            X, Y = me["x"] + ox, me["y"] + oy
            if all(_at(obs, "grid", X + dx, Y + dy) in (ground_ch - LIQUID_CH) | {YOU} for dx, dy in plan.clear) \
                    and flat(obs, X, Y, plan):
                d = math.hypot(ox + sx, oy + sy)
                if best is None or d < best[0]:
                    best = (d, (X, Y))
    return best and best[1]


def laid(obs, X, Y, piece) -> bool:
    if piece == "roof":
        return _at(obs, "roofs", X, Y) == "1"
    ch = _at(obs, "grid", X, Y)
    if piece == "path":
        return ch == PATH_CH
    return ch in WALL_CH if piece == "wall" else ch in DOOR_CH if piece == "door" else ch == CH.get(piece)


def next_step(obs, social, *, ground_ch, tool_for, dirs):
    """The home's next tool call (tool, args), or None."""
    if social is None:
        return None
    plan = plan_of(social)
    if getattr(social, "home", None) is None:
        social.home = site(obs, ground_ch, plan)
        if social.home is None:
            return None
    again = lambda: next_step(obs, social, ground_ch=ground_ch, tool_for=tool_for, dirs=dirs)    # noqa: E731
    hx, hy = social.home
    me, inv = obs["self"], obs["self"].get("inventory", {})
    cx, cy = hx + plan.ground[0][0], hy + plan.ground[0][1]
    if _at(obs, "grid", cx, cy) is None:                     # (out of view: walk home, a view at a time)
        dx, dy = cx - me["x"], cy - me["y"]
        d = math.hypot(dx, dy)
        if d > FAR:                                          # (far off - fallen and back at spawn: a home here)
            social.home = None
            return again()
        k = (VIEW_R - 1) / d
        return "move_to", {"x": me["x"] + round(dx * k), "y": me["y"] + round(dy * k)}
    is_laid = laid_by(obs, social, plan)
    if not flat(obs, hx, hy, plan):                          # (dug into since: the game refuses a wall by a pit)
        pit = _pit(obs, hx, hy, plan)
        earth = next((k for k in EARTH if inv.get(k)), None)
        if not any(is_laid(c) for c in plan.cells):
            social.home = None                               # (nothing laid on it yet: another spot)
            return again() if site(obs, ground_ch, plan) else None
        if pit is not None and earth:                        # (a home begun is kept and mended: the pit filled, as a
            (px, py), (sx, sy) = pit                         # player fills one, from where it stands, with earth from
            if (me["x"], me["y"]) != (hx + sx, hy + sy):     # the pack; without earth it goes on with what still goes
                return "move_to", {"x": hx + sx, "y": hy + sy}       # down - r63 left a closed room for a pit 4 tiles off)
            return "place", {"item": earth, "x": px, "y": py}
    on_land = {(x, y): p for x, y, z, p in plan.cells if z == 0 and p not in CELL_BLOCKS}
    for dx, dy in sorted(plan.clear, key=lambda q: (q[1], q[0])):      # (the ground cleared first, as a player clears
        X, Y = hx + dx, hy + dy                                           # it: a tree, a stump, a flower grown since)
        ch, piece = _at(obs, "grid", X, Y), on_land.get((dx, dy))
        if ch == ITEMS:                                      # (a pile on it - what was dug there dropped: taken up)
            return "move_to", {"x": X, "y": Y}
        if ch == YOU and any(i.get("dx") == 0 and i.get("dy") == 0 for i in obs.get("items") or []):
            return "pick_up", {}
        if ch in BLOCK_CH and not (piece and laid(obs, X, Y, piece)) and ch not in WALL_CH | DOOR_CH:
            d = next((k for k, v in dirs.items() if v == (X - me["x"], Y - me["y"])), None)
            if d is None:                                    # (to a tile beside it: a flower is walked onto, and
                by = next(((X + i, Y + j) for i, j in dirs.values()            # the tile one stands on shows it, not
                           if _at(obs, "grid", X + i, Y + j) in ground_ch), (X, Y))   # the flower)
                return "move_to", {"x": by[0], "y": by[1]}
            w = tool_for(obs, d)
            return "dig", {"dir": d, **({"with": w} if w else {})}
    land = level_at(obs, hx + plan.ground[0][0], hy + plan.ground[0][1])
    open_ = lambda s: _at(obs, "grid", hx + s[0], hy + s[1]) in ground_ch | {YOU}       # noqa: E731
    up = lambda dz: (t := plan.ups.get(dz)) is None or is_laid((t[0], t[1], 0, "stairs"))  # noqa: E731
    for cell in plan.cells:
        dx, dy, dz, piece = cell
        X, Y = hx + dx, hy + dy
        if is_laid(cell) or not plan.held(cell, is_laid) or (dz == 0 and (me["x"], me["y"]) == (X, Y) and piece not in CELL_BLOCKS):
            continue                                         # (under its feet: the tile shows it, not what is laid)
        if dz == 0 and piece not in CELL_BLOCKS and _at(obs, "grid", X, Y) not in ground_ch:
            continue                                         # (someone or something on it: the next piece)
        if _item(piece, inv) is None:
            step = next((s for w in (WALLS if piece == "wall" else (piece,)) if (s := craft(obs, w, ground_ch, tool_for, dirs))), None)
            if step is not None:
                return step
            continue
        stand = plan.stand_for(cell, open_, up)
        if stand is None:
            continue
        run = _run(obs, hx, hy, ground_ch, inv, plan, stand, is_laid, land)
        if not run:
            continue                                         # (a roof waiting on its ring: the next piece)
        sx, sy, sz = hx + stand[0], hy + stand[1], land + stand[2]        # (walk to where it is laid from first: one
        here = (me["x"], me["y"]) == (sx, sy) and (not stand[2] or me.get("z") == sz)    # pick walks there and lays
        go = {} if here else {"go": {"x": sx, "y": sy, **({"z": sz} if stand[2] else {})}}   # all in reach)
        return "ring", {"pieces": run, **go}
    return None


def level_at(obs, X, Y) -> int:
    """The land's level at a tile, from the observation's heights (none sent: all the low ground, 0)."""
    from .actions import LEVEL
    return LEVEL.get(_at(obs, "heights", X, Y), 0) if obs.get("heights") else 0


def laid_by(obs, social, plan):
    """Is the cell laid: on the land, as the observation shows it; a block a storey over the land, as its roofs show;
    higher up, or a flight of stairs, as the game said when it was laid (social.up_laid: the view is the land's)."""
    hx, hy = social.home
    land = level_at(obs, hx + plan.ground[0][0], hy + plan.ground[0][1])
    up = getattr(social, "up_laid", set())

    def is_laid(cell):
        dx, dy, dz, p = cell
        X, Y = hx + dx, hy + dy
        if p == "stairs":
            return (X, Y, "stairs") in up
        if dz == 0:
            return laid(obs, X, Y, p)
        if p in CELL_BLOCKS and dz == STOREY and _at(obs, "roofs", X, Y) == "1":
            return True
        return (X, Y, land + dz) in up
    return is_laid


def _item(piece, inv):
    """What the pack lays the piece with: a wall its most plentiful wall, else the piece itself; None when it lacks it."""
    if piece == "wall":
        return max((w for w in WALLS if inv.get(w)), key=inv.get, default=None)
    return piece if inv.get(piece) else None


def _run(obs, hx, hy, ground_ch, inv, plan, stand, is_laid, land=0) -> list:
    """The pieces it can lay now from `stand`, in the plan's order, as far as the pack goes: [(x, y, item[, z])]. One
    pick of build lays them all (rl/verbs.py split), so a pick pays for what it builds. A roof a storey over the land
    goes as the game lays one by hand (no z: a storey over the ground one stands on); any other block, and a thing
    up a storey, at its height."""
    left, out, going = dict(inv), [], set()
    now = lambda c: is_laid(c) or c in going                                  # noqa: E731
    for cell in plan.cells:
        dx, dy, dz, piece = cell
        X, Y = hx + dx, hy + dy
        if now(cell) or not plan.reaches(stand, cell) or not plan.held(cell, now):
            continue
        if dz == 0 and piece not in CELL_BLOCKS and _at(obs, "grid", X, Y) not in ground_ch:
            continue
        if piece in LAND_IN_CELL and _at(obs, "grid", X, Y) not in ground_ch | {YOU}:
            continue                                         # (earth over a thing grown on the land: the game refuses it)
        item = _item(piece, left)
        if item is None:
            continue
        left[item] -= 1
        going.add(cell)
        out.append((X, Y, item) if dz == 0 or (piece == "roof" and dz == STOREY) else (X, Y, item, land + dz))
    return out
