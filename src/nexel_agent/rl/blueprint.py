"""A home drawn as a tile grid, top-down, north up: what the architect (rl/architect.py) writes and the build verb lays
(#66). One character a tile:

    #  wall          D  door         w  window (a fence)      .  floor (inside, under the roof)
    b  bed           f  campfire     c  chest                 t  table       h  chair      k  workbench
    =  path (cobble, outside)        (space or ,)  outside, left as it is

The roof is not drawn: it goes over the walls and all they hold, and one tile past them (the overhang the looks
judge asks for). A grid is judged as the game would read it once built (rl/judge.py: v1 verdict, v2 sense, v3 looks),
stamped on cleared grass in a sim world, so a design is known good or not before a block is laid."""

from . import judge

LEGEND = {"#": "wall", "D": "door", "w": "fence", "b": "bed", "f": "campfire", "c": "chest", "t": "table",
          "h": "chair", "k": "workbench", "=": "path", ".": "floor"}
SHELL = ("wall", "door", "fence")          # laid first: the ring, then what stands in the room, the path, the roof
MOST = 13                                  # tiles a side, at the most: a home, not a hall
WALL = "wood_wall"                         # what a wall is stamped as when judged (the material is the pack's when built)
# v3's window check cannot pass with v1's closed one: the game's shelter counts a window (a fence) as a gap. Left out
# of a drawing's grade, until the game has a window that closes a room.
CANNOT = ("v3.windows",)


def parse(text: str) -> list:
    """The grid's rows as strings, the blank lines and a code fence around them dropped; ValueError when it is no
    drawing (an unknown character, too big, nothing built)."""
    rows = [r.rstrip() for r in text.strip("\n").splitlines() if r.strip() and not r.strip().startswith("```")]
    if not rows:
        raise ValueError("no grid")
    if len(rows) > MOST or max(map(len, rows)) > MOST:
        raise ValueError(f"too big: at most {MOST} by {MOST}")
    bad = {ch for r in rows for ch in r if ch not in LEGEND and ch not in " ,"}
    if bad:
        raise ValueError(f"unknown characters: {''.join(sorted(bad))}")
    if not any(ch == "#" for r in rows for ch in r):
        raise ValueError("no walls")
    return rows


def cells(rows) -> dict:
    """{(dx, dy): piece} from the grid's top-left, floor and outside left out."""
    return {(x, y): LEGEND[ch] for y, r in enumerate(rows) for x, ch in enumerate(r) if ch in LEGEND and ch != "."}


def roof(rows) -> list:
    """The roof's tiles: the box round the walls, one past each side."""
    walls = [p for p, k in cells(rows).items() if k in SHELL]
    xs, ys = [x for x, _ in walls], [y for _, y in walls]
    return [(x, y) for y in range(min(ys) - 1, max(ys) + 2) for x in range(min(xs) - 1, max(xs) + 2)]


def pieces(rows) -> list:
    """(dx, dy, piece) in the order they are laid: the shell, the furniture, the path, then the roof."""
    c = cells(rows)
    order = [p for p, k in c.items() if k in SHELL] + [p for p, k in c.items() if k not in SHELL and k != "path"] \
        + [p for p, k in c.items() if k == "path"]
    return [(x, y, c[(x, y)]) for x, y in order] + [(x, y, "roof") for x, y in roof(rows)]


def inside_point(rows):
    """A floor tile of the grid, from its top-left (where the judge reads the room from), or None."""
    return next(((x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "."), None)


def stamp(sim, rows, x, y):
    """The grid built at (x, y) (its top-left) in `sim`, on grass cleared round it, as if every piece were laid."""
    from nexel_worlds.sim.storeys import STOREY
    h, w = len(rows), max(map(len, rows))
    lv = sim.world.level(x, y)
    for i in range(-3, w + 3):
        for j in range(-3, h + 3):
            sim.world.set(x + i, y + j, ground="grass", block="none", level=lv)
    for dx, dy, piece in pieces(rows):
        X, Y = x + dx, y + dy
        if piece == "roof":
            sim.set_block(X, Y, lv + STOREY, "roof")
        elif piece == "path":
            sim.world.set(X, Y, ground="cobble")
        else:
            sim.world.set(X, Y, block=WALL if piece == "wall" else piece)


OF = len(judge.CHECKS) + len(judge.SENSE) + len(judge.LOOKS) - len(CANNOT)


def judged(text: str) -> dict:
    """The drawing judged: {"ok", "error"} when it is no drawing, else {"ok": True, "grade", "of", "failed":
    [check names, prefixed v1/v2/v3], "told": the failures in words, "rows"}."""
    from .core import SimEnv
    try:
        rows = parse(text)
    except ValueError as e:
        return {"ok": False, "error": str(e), "grade": 0, "of": OF,
                "failed": [], "told": str(e)}
    p = inside_point(rows)
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    x, y = a.x - 20, a.y - 20                                # (away from where it stands: it would be in the room)
    stamp(sim, rows, x, y)
    if p is None:
        return {"ok": True, "grade": 0, "of": OF, "rows": rows,
                "failed": ["v1.closed"], "told": "no floor ('.') inside the walls"}
    px, py = x + p[0], y + p[1]
    out = {"v1": judge.verdict(sim.world, px, py), "v2": judge.sense(sim.world, px, py),
           "v3": judge.looks(sim.world, px, py)}
    failed = [f"{k}.{c}" for k, v in out.items() for c in v["failed"] if f"{k}.{c}" not in CANNOT]
    return {"ok": True, "grade": OF - len(failed), "of": OF,
            "failed": failed, "told": told(out), "rows": rows, "by": {k: v["grade"] for k, v in out.items()}}


WHY = {"closed": "the walls do not close a room (a gap, or a tile outside reached from the floor)",
       "door": "no door in the walls", "roofed": "not roofed", "lit": "no light inside (a campfire)",
       "bed": "no bed inside", "room_enough": "too little floor inside",
       "chest": "no chest inside", "station": "no workbench or campfire in the home or by its door",
       "seat": "no table or chair", "reachable": "something inside cannot be reached on foot from the door",
       "door_out": "the door does not open onto open ground", "fit": "the inside is too small or too big for one (6 to 40 floor tiles)",
       "tidy": "land raised inside (a block of earth put down)", "light_by_bed": "no light near the bed",
       "material": "mixed wall materials", "rectangles": "the walls are not a clean rectangle (or a few)",
       "straight": "walls with one-tile jogs", "roof": "the roof stops at the walls",
       "symmetric": "the door is off the middle of its wall", "windows": "no window (w) in the walls (but a window leaves the room open to the game: closed fails)",
       "path": "no path (=) to the door", "no_stray": "a piece left standing outside the walls"}


def told(out: dict) -> str:
    fails = [WHY.get(c, c) for k, v in out.items() for c in v["failed"] if f"{k}.{c}" not in CANNOT]
    return "; ".join(dict.fromkeys(fails)) or "nothing failed"
