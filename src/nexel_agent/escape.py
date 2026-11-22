"""Getting out when no walk works from here (gatsu0, 2026-11-22: swimming in a river at the foot of cliffs 10 layers
high, every move_to to the land failed). Player actions only: swim to the nearest bank low enough to walk up,
else swim on along the water to see more, else cut a stair into the lowest wall (dig the block at head height in
front, jump onto the one under it).
"""

from .policy import WALK_CHARS
from .tiles import DIRS

WATER = set("~(=)")
STUCK_AFTER = 3                                  # no_path in a row before the body calls itself stuck
SWIMS = 3                                        # views' worth of swimming for a low bank before cutting a stair


def level(ch: str) -> int:
    """A heights char back to its layer (observe.height_char): 0-9, A.. = 10.., a.. = -1.."""
    if ch.isdigit():
        return int(ch)
    return ord(ch) - 55 if ch.isupper() else 96 - ord(ch)


def climb_reach(obs) -> dict:
    """Tiles in view a body can get to from here, walking or swimming, up one layer at a time (down any)."""
    grid, heights = obs["grid"], obs.get("heights")
    r = len(grid) // 2
    h = (lambda i, j: level(heights[j][i])) if heights else (lambda i, j: 0)
    ok = WALK_CHARS | WATER | {"@"}
    seen, frontier = {(0, 0): 0}, [(0, 0)]
    while frontier:
        nxt = []
        for cx, cy in frontier:
            for dx, dy in DIRS.values():
                q = (cx + dx, cy + dy)
                i, j = r + q[0], r + q[1]
                if q in seen or not (0 <= i < len(grid) and 0 <= j < len(grid)) or grid[j][i] not in ok:
                    continue
                if h(i, j) - h(r + cx, r + cy) <= 1:
                    seen[q] = seen[(cx, cy)] + 1
                    nxt.append(q)
        frontier = nxt
    return seen


def stuck(obs, memory) -> bool:
    """No walk works from here: moves keep failing for want of a path, or in water no land in view is reachable."""
    me = obs["self"]
    if memory.get("climb_to") is not None:       # cutting a stair: on until up at the wall's top
        if me.get("z", 0) < memory["climb_to"] and memory.get("climb_steps", 0) < 40:
            return True
        for k in ("climb_to", "stair_dir", "stair", "swims", "climb_steps"):
            memory.pop(k, None)
    if memory.get("no_path", 0) >= STUCK_AFTER:
        return True
    if me.get("swimming"):
        r = len(obs["grid"]) // 2
        return not any(obs["grid"][r + q[1]][r + q[0]] in WALK_CHARS for q in climb_reach(obs))
    memory.pop("swims", None)
    return False


def way_out(obs, memory) -> tuple[str, dict, str] | None:
    me, grid = obs["self"], obs["grid"]
    r = len(grid) // 2
    x, y = me["x"], me["y"]
    z = me.get("z", 0)
    if memory.get("stair"):                      # cutting a stair: the next cut, or the jump onto it
        return memory["stair"].pop(0) if len(memory["stair"]) > 1 else memory.pop("stair")[0]
    tried = memory.get("escape_avoid", set())
    reach = {q: n for q, n in climb_reach(obs).items() if (x + q[0], y + q[1]) not in tried}
    land = [(n, q) for q, n in reach.items() if q != (0, 0) and grid[r + q[1]][r + q[0]] in WALK_CHARS]
    if land:
        n, (dx, dy) = min(land)
        return "move_to", {"x": x + dx, "y": y + dy}, "getting out: to the nearest bank I can walk up"
    edge = [q for q, n in reach.items() if max(abs(q[0]), abs(q[1])) >= r - 2]
    if edge and memory.get("swims", 0) < SWIMS:  # swim on, the same way as before, a few views' worth
        hd = memory.get("swim_heading") or edge[0]
        dx, dy = max(edge, key=lambda q: q[0] * hd[0] + q[1] * hd[1])
        memory["swim_heading"], memory["swims"] = (dx, dy), memory.get("swims", 0) + 1
        return "move_to", {"x": x + dx, "y": y + dy}, "getting out: swimming on to find a low bank"
    heights = obs.get("heights")
    if not heights:
        return None
    walls = [(level(heights[r + dy][r + dx]), d) for d, (dx, dy) in DIRS.items()
             if grid[r + dy][r + dx] not in WATER]
    if not walls:                                # open water all round: go to the water's edge first
        side = [(n, q) for q, n in reach.items() if any(grid[r + q[1] + b][r + q[0] + a] not in WATER | {"@"}
                for a, b in DIRS.values() if abs(q[0] + a) <= r and abs(q[1] + b) <= r)]
        if not side:
            return None
        _, (dx, dy) = min(side)
        return "move_to", {"x": x + dx, "y": y + dy}, "getting out: to the foot of the wall"
    top, d = min(walls)                          # cut a stair into the lowest wall: head-height block out, jump on
    if memory.get("stair_dir") in [w[1] for w in walls]:
        d = memory["stair_dir"]                  # (straight on into the same wall, a layer a time)
    memory["stair_dir"] = d
    memory.setdefault("climb_to", top)
    memory["climb_steps"] = memory.get("climb_steps", 0) + 1
    memory["stair"] = [("dig", {"dir": d, "z": z + 2}, "getting out: cutting a step into the wall"),
                       ("jump", {"dir": d}, "climbing out: onto the step")]
    return "dig", {"dir": "up"}, "getting out: room over my head for the climb"


def escape_failed(memory, tool, args):
    """A way out that did not work is not tried again (from this view). (Nothing over the head to dig: fine.)"""
    if tool == "dig" and args.get("dir") == "up":
        return
    memory.pop("stair", None)
    if tool == "move_to":
        memory.setdefault("escape_avoid", set()).add((args.get("x"), args.get("y")))


def stuck_line(obs) -> str:
    me = obs["self"]
    where = "in water at the foot of a cliff" if me.get("swimming") else "in a hole or walled in"
    return (f"You are stuck: no path from here to the land ({where}). Get out first: swim along the water to a low "
            "bank, or dig the block in front at z = your z + 2 and jump onto it, a layer at a time.")
