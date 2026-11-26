"""The `build` verb (#28): a blueprint's next step, as one game tool call - the "how" of a home, never the "when" (the
policy chooses build or anything else, and when to come back to it). A hut: a 3 x 3 inside, walled all round with a
door on the south side, a fire and a bed in two corners, a roof over all of it, laid the way a player does it,
standing in the middle where every piece is in reach (tiles.REACH "use", the lead counted).

Read from the game: what closes a room (Simulation.ENCLOSE, a wall the solid ones, a door the others), what each
piece is made of (tiles.RECIPES, through pursue: a piece the pack lacks is made, or what it takes is gathered), what
is laid already (the observation's grid, and its roofs). Its one memory is where the home is: social.home, chosen once,
on flat open ground near where it stands. None: no step to take now (nowhere flat in view, nothing it can lay or make)."""

import math

from nexel_worlds.sim.constants import VIEW_R
from nexel_worlds.sim.observe import YOU
from nexel_worlds.tiles import BLOCKS

from .judge import ENCLOSE
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
BLOCK_CH = {b.char for b in BLOCKS if b.name != "none" and b.solid and b.hardness > 0}   # (what stands in the way and can be dug)


def blueprint() -> list:
    """(dx, dy, piece) from the inside's top-left, in the order it is laid: walls, door, furniture, then the roof."""
    ring = [(dx, dy) for dy in range(-1, H + 1) for dx in range(-1, W + 1) if dx in (-1, W) or dy in (-1, H)]
    return ([(dx, dy, "wall") for dx, dy in ring if (dx, dy) != DOOR] + [(*DOOR, "door")]
            + [(dx, dy, p) for dx, dy, p in FURNITURE]
            + [(dx, dy, "roof") for dy in range(-1, H + 1) for dx in range(-1, W + 1)])


PLAN = blueprint()


def _at(obs, key, X, Y):
    me, rows = obs["self"], obs.get(key)
    gx, gy = X - me["x"] + VIEW_R, Y - me["y"] + VIEW_R
    if not rows or not (0 <= gy < len(rows) and 0 <= gx < len(rows[gy])):
        return None
    return rows[gy][gx]


def flat(obs, X, Y) -> bool:
    """The hut whose inside's top-left is (X, Y) stands on one level, clear of a hill's rim: no lower tile beside it,
    diagonals too (the game builds nothing there: world.rim). A pit dug in it since makes it no site."""
    from .actions import LEVEL
    if not obs.get("heights"):
        return True                                          # (no relief in view at all)
    lv = LEVEL.get(_at(obs, "heights", X + 1, Y + 1))
    if lv is None:
        return False
    for dx in range(-2, W + 2):
        for dy in range(-2, H + 2):
            h = LEVEL.get(_at(obs, "heights", X + dx, Y + dy), -99)
            if h < lv or (-1 <= dx <= W and -1 <= dy <= H and h != lv):
                return False
    return True


def site(obs, ground_ch) -> tuple | None:
    """The inside's top-left of the nearest spot in view where the whole hut stands on flat, open ground."""
    me = obs["self"]
    best = None
    for oy in range(-VIEW_R + 2, VIEW_R - H - 1):
        for ox in range(-VIEW_R + 2, VIEW_R - W - 1):
            X, Y = me["x"] + ox, me["y"] + oy
            if all(_at(obs, "grid", X + dx, Y + dy) in ground_ch | {YOU} for dx, dy, p in PLAN if p != "roof") and flat(obs, X, Y):
                d = math.hypot(ox + 1, oy + 1)
                if best is None or d < best[0]:
                    best = (d, (X, Y))
    return best and best[1]


def laid(obs, X, Y, piece) -> bool:
    if piece == "roof":
        return _at(obs, "roofs", X, Y) == "1"
    ch = _at(obs, "grid", X, Y)
    return ch in WALL_CH if piece == "wall" else ch in DOOR_CH if piece == "door" else ch == CH.get(piece)


def next_step(obs, social, *, ground_ch, tool_for, dirs):
    """The hut's next tool call (tool, args), or None."""
    if social is None:
        return None
    if getattr(social, "home", None) is None:
        social.home = site(obs, ground_ch)
        if social.home is None:
            return None
    hx, hy = social.home
    me, inv = obs["self"], obs["self"].get("inventory", {})
    cx, cy = hx + W // 2, hy + H // 2
    if _at(obs, "grid", cx, cy) is None:                     # (out of view: walk home, a view at a time)
        dx, dy = cx - me["x"], cy - me["y"]
        d = math.hypot(dx, dy)
        if d > FAR:                                          # (far off - fallen and back at spawn: a home here)
            social.home = None
            return next_step(obs, social, ground_ch=ground_ch, tool_for=tool_for, dirs=dirs)
        k = (VIEW_R - 1) / d
        return "move_to", {"x": me["x"] + round(dx * k), "y": me["y"] + round(dy * k)}
    if not flat(obs, hx, hy):                                # (dug into since: the game would refuse it - another spot)
        social.home = None
        return next_step(obs, social, ground_ch=ground_ch, tool_for=tool_for, dirs=dirs) if site(obs, ground_ch) else None
    for dx, dy, piece in PLAN:                               # (the ground cleared first, as a player clears it:
        X, Y = hx + dx, hy + dy                              # a tree, a stump or a bush grown on the spot since)
        ch = _at(obs, "grid", X, Y)
        if piece != "roof" and ch in BLOCK_CH and not laid(obs, X, Y, piece) and ch not in WALL_CH | DOOR_CH:
            d = next((k for k, v in dirs.items() if v == (X - me["x"], Y - me["y"])), None)
            if d is None:
                return "move_to", {"x": X, "y": Y}                # (the server walks beside it)
            w = tool_for(obs, d)
            return "dig", {"dir": d, **({"with": w} if w else {})}
    for dx, dy, piece in PLAN:
        X, Y = hx + dx, hy + dy
        if laid(obs, X, Y, piece):
            continue
        if piece != "roof" and _at(obs, "grid", X, Y) not in ground_ch:
            continue                                         # (someone or something on it: the next piece)
        item = max((w for w in WALLS if inv.get(w)), key=inv.get, default=None) if piece == "wall" else piece if inv.get(piece) else None
        if item is None:
            step = next((s for w in (WALLS if piece == "wall" else (piece,)) if (s := craft(obs, w, ground_ch, tool_for, dirs))), None)
            if step is not None:
                return step
            continue
        go = {} if (me["x"], me["y"]) == (cx, cy) else {"go": {"x": cx, "y": cy}}   # (walk to the middle first: one
        return "ring", {"pieces": _run(obs, hx, hy, ground_ch, inv), **go}         # pick walks there and lays all)
    return None


def _run(obs, hx, hy, ground_ch, inv) -> list:
    """The pieces it can lay now from the middle, in the plan's order, as far as the pack goes: [(x, y, item)]. One
    pick of build lays them all (rl/verbs.py split, as ring does), so a pick pays for what it builds."""
    left, out = dict(inv), []
    for dx, dy, piece in PLAN:
        X, Y = hx + dx, hy + dy
        if laid(obs, X, Y, piece) or (piece != "roof" and _at(obs, "grid", X, Y) not in ground_ch):
            continue
        item = max((w for w in WALLS if left.get(w)), key=left.get, default=None) if piece == "wall" else piece if left.get(piece) else None
        if item is None:
            continue
        left[item] -= 1
        out.append((X, Y, item))
    return out
