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
from nexel_worlds.tiles import GROUNDS, TILL_TOOL
from nexel_worlds.tiles import BLOCKS

from .judge import ENCLOSE
from .social import laid_as
from .plan import BLOCKS as CELL_BLOCKS
from .site import _at, _pit, flat, no_way, site, tried, way_out
from .plan import Plan, drawn, of_pieces  # noqa: F401  (drawn: the grid's plan, for its callers)
from .pursue import craft

W, H, DOOR = 3, 3, (1, 3)                                   # the inside; the door in the south wall's middle
FURNITURE = ((0, 0, "campfire"), (W - 1, 0, "bed"))         # a corner each
FAR = 240                   # a home further off is given up (fallen, back at spawn); a trip for wood or stone goes 100+
_SOLID, CH = {b.name: b.solid for b in BLOCKS}, {b.name: b.char for b in BLOCKS}
WALLS, DOORS = tuple(k for k in ENCLOSE if _SOLID[k]), tuple(k for k in ENCLOSE if not _SOLID[k])
WALL_CH, DOOR_CH = {CH[k] for k in WALLS}, {CH[k] for k in DOORS}
DIRS4 = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
TIER = {"wood_pickaxe": 1, "pickaxe": 2, "iron_pickaxe": 3, "titanium_pickaxe": 4}      # (the best digs fastest)
TILLED_CH = {g.char for g in GROUNDS if g.name in ("tilled", "wet_tilled")}
SOWN_CH = {CH[k] for k in ("sprout", "crop", "ripe_crop")}
PATH_CH = next(g.char for g in GROUNDS if g.name == "cobble")                           # (what the path item lays)
NATURAL = ("tree", "fruit_tree", "bare_fruit_tree", "stump", "bush", "berry_bush", "bare_bush", "thicket", "rock",
           "mushroom", "flower")                     # (what grows or lies on the land: cleared off a site, as a player
CLEARABLE = frozenset(CH[k] for k in NATURAL if k in CH)    # clears it; never what someone made - a chest, a bed, crops)


def blueprint() -> list:
    """(dx, dy, piece) from the inside's top-left, in the order it is laid: walls, door, furniture, then the roof."""
    ring = [(dx, dy) for dy in range(-1, H + 1) for dx in range(-1, W + 1) if dx in (-1, W) or dy in (-1, H)]
    return ([(dx, dy, "wall") for dx, dy in ring if (dx, dy) != DOOR] + [(*DOOR, "door")]
            + [(dx, dy, p) for dx, dy, p in FURNITURE]
            + [(dx, dy, "roof") for dy in range(-1, H + 1) for dx in range(-1, W + 1)])


PLAN = blueprint()


HUT = of_pieces(PLAN, ((W // 2, H // 2),))


def plan_of(social) -> Plan:
    return getattr(social, "plan", None) or HUT


def laid(obs, X, Y, piece) -> bool:
    if piece == "roof":
        return _at(obs, "roofs", X, Y) == "1"
    ch = _at(obs, "grid", X, Y)
    if piece == "path":
        return ch == PATH_CH
    if piece == "till":                                     # (tilled, or sown already)
        return ch in TILLED_CH or ch in SOWN_CH
    if piece == "seed":
        return ch in SOWN_CH
    return ch in WALL_CH if piece == "wall" else ch in DOOR_CH if piece == "door" else ch == CH.get(piece)


def next_step(obs, social, *, ground_ch, tool_for, dirs):
    """The home's next tool call (tool, args), or None."""
    if social is None:
        return None
    plan = plan_of(social)
    if getattr(social, "home", None) is None:
        social.home = site(obs, ground_ch, plan, frozenset() if plan is HUT else CLEARABLE)
        if social.home is None:
            return _look_further(obs, social, plan)
        social.search = None
    again = lambda: next_step(obs, social, ground_ch=ground_ch, tool_for=tool_for, dirs=dirs)    # noqa: E731
    hx, hy = social.home
    me, inv = obs["self"], obs["self"].get("inventory", {})
    if (out := way_out(obs, ground_ch, own(social), dirs, tool_for)) is not None:
        return out
    cx, cy = min(((hx + x, hy + y) for x, y in plan.clear), key=lambda p: abs(p[0] - me["x"]) + abs(p[1] - me["y"]))
    if _at(obs, "grid", cx, cy) is None:                     # (none of it in view: walk home, a view at a time, to its
        dx, dy = cx - me["x"], cy - me["y"]                  # nearest tile - a compound is longer than a view)
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
    on_land = {(x, y): p for x, y, z, p in plan.cells if z <= 0 and p not in (*CELL_BLOCKS, "dig")}   # (a dug floor's too)
    near_me = lambda q: (abs(hx + q[0] - me["x"]) + abs(hy + q[1] - me["y"]), q[1], q[0])    # noqa: E731
    for dx, dy in sorted(plan.clear, key=near_me):          # (the ground cleared first, the nearest first, as a player clears
        X, Y = hx + dx, hy + dy                                           # it: a tree, a stump, a flower grown since)
        ch, piece = _at(obs, "grid", X, Y), on_land.get((dx, dy))
        if ch == ITEMS:                                      # (a pile on it - what was dug there dropped: taken up)
            return "move_to", {"x": X, "y": Y}
        if ch == YOU and any(i.get("dx") == 0 and i.get("dy") == 0 for i in obs.get("items") or []):
            return "pick_up", {}
        if ch in CLEARABLE and not (piece and laid(obs, X, Y, piece)):
            d = next((k for k, v in dirs.items() if v == (X - me["x"], Y - me["y"])), None)
            if d is not None and not tried(social, X, Y, (me["x"], me["y"]), count=not me.get("action")):  # (a dig under way: no try)
                return "dig", {"dir": d, **({"with": w} if (w := tool_for(obs, d)) else {})}
            by = next(((X + i, Y + j) for i, j in dirs.values() if _at(obs, "grid", X + i, Y + j) in ground_ch  # (to a tile
                       and not tried(social, X, Y, (X + i, Y + j), count=False)), None if d else (X, Y))  # beside it; a
            if by is None:                                   # flower is walked onto; refused from every side: left)
                continue
            return "move_to", {"x": by[0], "y": by[1]}
    land = home_level(obs, social, plan)
    open_ = lambda s: _at(obs, "grid", hx + s[0], hy + s[1]) in ground_ch | {YOU}       # noqa: E731
    underfoot = None
    for cell in plan.cells:
        dx, dy, dz, piece = cell
        X, Y = hx + dx, hy + dy
        if is_laid(cell) or not plan.held(cell, is_laid):
            continue
        if dz <= 0 and (me["x"], me["y"]) == (X, Y) and piece not in CELL_BLOCKS:
            underfoot = underfoot or cell                    # (under its feet: the tile shows it, not what is laid)
            continue
        if dz <= 0 and piece not in (*CELL_BLOCKS, "dig") and _at(obs, "grid", X, Y) not in ground_ch:
            continue                                         # (someone or something on it: the next piece)
        if piece != "dig" and (_item(piece, inv) is None or inv.get(piece, 0) < getattr(plan, "flights", {}).get((dx, dy), 0)):
            if piece == "till" and (step := craft(obs, TILL_TOOL, ground_ch, tool_for, dirs, blocked=no_way(social), keep=own(social))):
                return step                                  # (no hoe: one made)
            step = next((s for w in (WALLS if piece == "wall" else (piece,)) if (s := craft(obs, w, ground_ch, tool_for, dirs, blocked=no_way(social), keep=own(social)))), None)
            if step is not None:
                return step
            continue
        stand = plan.stand_for(cell, open_, is_laid)
        if stand is None:
            continue
        run = _run(obs, hx, hy, ground_ch, inv, plan, stand, is_laid, land)
        if not run:
            continue                                         # (a roof waiting on its ring: the next piece)
        sx, sy, sz = hx + stand[0], hy + stand[1], land + stand[2]        # (walk to where it is laid from first: one
        here = (me["x"], me["y"]) == (sx, sy) and (not stand[2] or me.get("z") == sz)    # pick walks there and lays
        go = {} if here else {"go": {"x": sx, "y": sy, **({"z": sz} if stand[2] else {})}}   # all in reach)
        return "ring", {"pieces": run, **go}
    if underfoot is not None and (st := plan.stand_for(underfoot, open_, is_laid)) is not None:
        return "move_to", {"x": hx + st[0], "y": hy + st[1]}     # (all else done: a step off to see, and lay it)
    far = [(hx + c[0], hy + c[1]) for c in plan.cells if _at(obs, "grid", hx + c[0], hy + c[1]) is None and not is_laid(c)]
    if far:                                                  # (what is left is out of view - the other end of a
        fx, fy = min(far, key=lambda p: abs(p[0] - me["x"]) + abs(p[1] - me["y"]))     # compound: a view nearer it)
        to = [(me["x"] + round((fx - me["x"]) * k), me["y"] + round((fy - me["y"]) * k)) for k in   # (the furthest
              (min(1, (VIEW_R - 2) * i / 8 / math.hypot(fx - me["x"], fy - me["y"])) for i in range(8, 0, -1))]  # open
        return "move_to", dict(zip("xy", next((p for p in to if _at(obs, "grid", *p) in ground_ch), to[0])))  # ground)
    return None


SEARCH = tuple((dx * k, dy * k) for k in (1, 2)                # (a view's walk each way in turn, then two views)
               for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)))


def _look_further(obs, social, plan):
    """No spot in view for a drawn build: a view's walk from where it began looking, a way at a time, a look again
    there, as a player goes looking for flat land; the next way once there with none, or stuck (the sea in the way).
    The hut, the body's own, is sited in view or not at all, as it was trained."""
    if plan is HUT:
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
    return None


def home_level(obs, social, plan) -> int:
    """The land's level at the home: read where it is in view, and kept (out of view it is not 0, and what was dug or
    laid up there is known by its height)."""
    hx, hy = social.home
    X, Y = hx + plan.ground[0][0], hy + plan.ground[0][1]
    kept = getattr(social, "home_at", None)
    if kept and kept[0] == (X, Y) and _at(obs, "grid", X, Y) is None:
        return kept[1]
    lv = level_at(obs, X, Y)
    if _at(obs, "grid", X, Y) is not None:
        social.home_at = ((X, Y), lv)
    return lv


def level_at(obs, X, Y) -> int:
    """The land's level at a tile, from the observation's heights (none sent: all the low ground, 0)."""
    from .actions import LEVEL
    return LEVEL.get(_at(obs, "heights", X, Y), 0) if obs.get("heights") else 0


def own(social) -> set:
    """The home's tiles in the world: dug by the builder only (live trial 5: 463 digs on the site between builds)."""
    if getattr(social, "home", None) is None:
        return set()
    return {(social.home[0] + x, social.home[1] + y) for x, y in plan_of(social).clear | {c[:2] for c in plan_of(social).cells}}


def laid_by(obs, social, plan):
    """Is the cell laid: on the land, as the observation shows it; a block a storey over the land, as its roofs show;
    higher up, or a flight of stairs, as the game said when it was laid (social.up_laid: the view is the land's)."""
    hx, hy = social.home
    land = home_level(obs, social, plan)
    up = getattr(social, "up_laid", set())
    if getattr(social, "seen_laid", None) is None:
        social.seen_laid = set()                             # (laid on the land, as last seen)
    seen = social.seen_laid

    def is_laid(cell):
        dx, dy, dz, p = cell
        X, Y = hx + dx, hy + dy
        if p == "stairs":
            return (X, Y, "stairs") in up
        if p == "dig":                                       # (dug: as the game said, or the floor seen lower)
            return (X, Y, land + dz, "dug") in up or (obs.get("heights") is not None and level_at(obs, X, Y) < land + dz)
        if dz <= 0:                                          # (out of view now: as it was when last seen)
            if _at(obs, "grid", X, Y) in (None, YOU):        # (under its feet too: the tile shows it, not what is
                return (X, Y, dz, p) in seen                 # laid there)
            if laid(obs, X, Y, p):
                seen.add((X, Y, dz, p))
                return True
            seen.discard((X, Y, dz, p))
            return False
        if p in CELL_BLOCKS and dz == STOREY and _at(obs, "roofs", X, Y) == "1":
            return True
        return (X, Y, land + dz, laid_as(p)) in up
    return is_laid


def _item(piece, inv):
    """What the pack lays the piece with: a wall its most plentiful wall, soil tilled with a hoe, else the piece itself;
    None when it lacks it."""
    if piece == "till":
        return TILL_TOOL if inv.get(TILL_TOOL) else None
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
        if piece == "dig":                                   # (a cell of the land, from the tile beside it)
            sx, sy = hx + stand[0], hy + stand[1]
            d = next((k for k, v in DIRS4.items() if v == (X - sx, Y - sy)), None)
            pick = max((k for k in left if "pickaxe" in k and left[k]), key=lambda k: TIER.get(k, 0), default=None)
            going.add(cell)
            out.append((X, Y, "dig", land + dz, d, pick))
            continue
        if dz <= 0 and piece not in CELL_BLOCKS and _at(obs, "grid", X, Y) not in ground_ch:
            continue
        if piece == "till":                                  # (the hoe stays in the pack: soil after soil)
            if left.get(TILL_TOOL):
                going.add(cell)
                out.append((X, Y, "till", TILL_TOOL))
            continue
        item = _item(piece, left)
        if item is None or left.get(item, 0) < getattr(plan, "flights", {}).get((dx, dy), 0):   # (a flight: a step each)
            continue
        left[item] -= 1
        going.add(cell)
        out.append((X, Y, item) if dz <= 0 or (piece == "roof" and dz == STOREY) else (X, Y, item, land + dz))
    return out
