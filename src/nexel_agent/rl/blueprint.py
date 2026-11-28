"""A home drawn as a tile grid, top-down, north up: what the architect (rl/architect.py) writes and the build verb lays
(#66). One character a tile:

    #  wall          D  door         .  floor (inside, under the roof)
    b  bed           f  campfire     c  chest                 t  table       h  chair      k  workbench
    =  path (cobble, outside)        (space or ,)  outside, left as it is

The roof is not drawn: it goes over the walls and all they hold, and one tile past them (the overhang the looks
judge asks for). A grid is judged as the game would read it once built (rl/judge.py: v1 verdict, v2 sense, v3 looks),
stamped on cleared grass in a sim world, so a design is known good or not before a block is laid."""

from . import judge

LEGEND = {"#": "wall", "D": "door", "b": "bed", "f": "campfire", "c": "chest", "t": "table",
          "h": "chair", "k": "workbench", "=": "path", ".": "floor"}
SHELL = ("wall", "door")          # laid first: the ring, then what stands in the room, the path, the roof
MOST = 13                                  # tiles a side, at the most: a home, not a hall
WALL = "wood_wall"                         # what a wall is stamped as when judged (the material is the pack's when built)


def parse(text: str) -> list:
    """The grid's rows as strings, the blank lines and a code fence around them dropped; ValueError when it is no
    drawing (an unknown character, too big, nothing built; "? why": a design that could not be drawn, architect.py)."""
    if text.startswith("? "):
        raise ValueError(text[2:])
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


OF = len(judge.CHECKS) + len(judge.SENSE) + len(judge.LOOKS)


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
    failed = [f"{k}.{c}" for k, v in out.items() for c in v["failed"]]
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
       "symmetric": "the door is off the middle of its wall", "path": "no path (=) to the door", "no_stray": "a piece left standing outside the walls"}


def told(out: dict) -> str:
    """The failures in words, with the judge's facts where they say which: the pieces not reached, the floor's size."""
    f = out.get("v2", {}).get("facts", {})
    exact = {"reachable": f"not reached on foot from the door: {', '.join(f.get('unreached') or [])} (each needs a free "
                          f"tile beside it, joined to the door by free tiles; furniture blocks the way)",
             "fit": f"{f.get('area')} tiles inside: 6 to 40 for a home"}
    fails = [exact.get(c) if k == "v2" and c in exact and f.get("closed") else WHY.get(c, c)
             for k, v in out.items() for c in v["failed"]]
    return "; ".join(dict.fromkeys(fails)) or "nothing failed"


# ---- a design as choices (the model's), drawn by code: the 30B draws no grid well (spaced tiles, gaps, no door; #66
# bench run 1: 0 of 10), it chooses well. {"width", "depth": the inside, "door": "north"/"south"/"east"/"west",
# "furniture": {piece: [x, y] from the inside's top-left}, "path": tiles}

SIDES = ("north", "south", "east", "west")
FURNITURE = {v: k for k, v in LEGEND.items() if v not in ("wall", "door", "path", "floor")}


def render(spec: dict) -> str:
    """The grid of a design; ValueError, told in words, when it cannot be drawn."""
    try:
        w, d, door = int(spec["width"]), int(spec["depth"]), str(spec.get("door", "south")).lower()
        things, path = dict(spec.get("furniture") or {}), int(spec.get("path", 2))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"a design needs width, depth, door and furniture ({e})") from None
    if not (2 <= w <= MOST - 2 and 2 <= d <= MOST - 2):
        raise ValueError(f"width and depth from 2 to {MOST - 2}")
    if door not in SIDES:
        raise ValueError(f"door is one of {', '.join(SIDES)}")
    path = max(0, min(path, 3))
    g = [["#"] * (w + 2) for _ in range(d + 2)]
    for y in range(d):
        for x in range(w):
            g[y + 1][x + 1] = "."
    placed = [(n, at) for n, ats in things.items()                # (two of a piece: a list of places)
              for at in (ats if isinstance(ats, list) and ats and isinstance(ats[0], list) else [ats])]
    for name, at in placed:
        if name not in FURNITURE:
            raise ValueError(f"no such furniture: {name} (one of {', '.join(FURNITURE)})")
        try:
            x, y = (int(v) for v in at)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: a place is [x, y]") from None
        if not (0 <= x < w and 0 <= y < d):
            raise ValueError(f"{name} at [{x}, {y}] is outside the {w} by {d} inside (x 0 to {w - 1}, y 0 to {d - 1})")
        if g[y + 1][x + 1] != ".":
            raise ValueError(f"{name} at [{x}, {y}]: another piece is there")
        g[y + 1][x + 1] = FURNITURE[name]
    mx, my = (w + 1) // 2, (d + 1) // 2                       # (the wall's middle: exact when the side is odd)
    at = {"north": (mx, 0), "south": (mx, d + 1), "west": (0, my), "east": (w + 1, my)}[door]
    g[at[1]][at[0]] = "D"
    step = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}[door]
    pad = {"north": (0, path), "south": (0, 0), "west": (path, 0), "east": (0, 0)}[door]   # (room for the path out)
    rows = [[" "] * pad[0] + r for r in [[" "] * len(g[0]) for _ in range(pad[1])] + g]
    x, y = at[0] + pad[0], at[1] + pad[1]
    for _ in range(path):
        x, y = x + step[0], y + step[1]
        while y >= len(rows):
            rows.append([" "] * len(rows[0]))
        while x >= len(rows[y]):
            rows[y].append(" ")
        rows[y][x] = "="
    return "\n".join("".join(r).rstrip() for r in rows)


def _all_reached(w, d, front, taken) -> bool:
    """From the tile inside the door, on foot over the free tiles: every piece beside one walked (judge.sense's rule)."""
    if front in taken:
        return False
    walked, stack = set(), [front]
    while stack:
        p = stack.pop()
        if p in walked:
            continue
        walked.add(p)
        stack += [q for q in _nbrs(p) if 0 <= q[0] < w and 0 <= q[1] < d and q not in taken and q not in walked]
    return all(any(q in walked for q in _nbrs(t)) for t in taken)


def _nbrs(p):
    return [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]


def tidied(spec: dict) -> dict:
    """The design with its furniture where it all fits: each piece by a wall, as a player sets it, where it was chosen
    or the nearest tile to it that keeps everything reached on foot from the door (the middle only when the walls are
    full). The model
    chooses what and roughly where; this, that one can walk to all of it. A piece with no such tile is left out."""
    w, d, door = int(spec["width"]), int(spec["depth"]), str(spec.get("door", "south")).lower()
    mx, my = (w + 1) // 2 - 1, (d + 1) // 2 - 1              # (where render puts the door, inside)
    front = {"north": (mx, 0), "south": (mx, d - 1), "west": (0, my), "east": (w - 1, my)}.get(door, (mx, d - 1))
    by_wall = lambda q: 0 if q[0] in (0, w - 1) or q[1] in (0, d - 1) else 1      # noqa: E731
    tiles = [(x, y) for y in range(d) for x in range(w) if (x, y) != front]
    taken, out = set(), {}
    placed = [(n, at) for n, ats in (spec.get("furniture") or {}).items()
              for at in (ats if isinstance(ats, list) and ats and isinstance(ats[0], list) else [ats])]
    for name, at in placed:
        try:
            x, y = (int(v) for v in at)
        except (TypeError, ValueError):
            x, y = 0, 0
        ok = [q for q in tiles if q not in taken and _all_reached(w, d, front, taken | {q})]
        if not ok:
            continue
        q = min(ok, key=lambda q: (by_wall(q), abs(q[0] - x) + abs(q[1] - y)))      # (where chosen, if by a wall)
        taken.add(q)
        out.setdefault(name, []).append(list(q))
    return {**spec, "furniture": out}
