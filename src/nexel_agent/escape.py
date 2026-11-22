"""Getting out when no walk works from here (gatsu0, 2026-11-22: swimming in a river at the foot of cliffs 10 layers
high, every move_to to the land failed). Player actions only, three routines in order: "bank" (walk to reachable
land that goes on past the view), "swim" (along the water, a heading kept and a target at the view's edge),
"stair" (cut into the lowest wall: dig the block at head height in front, jump onto the one under it). Each keeps
its target until reached or failed; the watchdog bans one that goes nowhere, and the next is tried.
"""

from .policy import WALK_CHARS
from .tiles import DIRS

WATER = set("~(=)")
STUCK_AFTER = 3                                  # no_path in a row before the body calls itself stuck
CLIMB = ("climb_to", "stair_dir", "stair", "climb_steps")


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


def at_border(q, r) -> bool:
    return max(abs(q[0]), abs(q[1])) >= r - 1


def walkable_out(obs, reach) -> list:
    """(steps, q) of reachable land tiles at the edge of the view: land that goes on beyond what is seen."""
    grid, r = obs["grid"], len(obs["grid"]) // 2
    return sorted((n, q) for q, n in reach.items() if at_border(q, r) and grid[r + q[1]][r + q[0]] in WALK_CHARS)


def stuck(obs, memory) -> bool:
    """No walk gets anywhere from here: moves keep failing for want of a path; or in water with no land to walk
    up onto; or on land in a closed pocket (nothing reachable reaches the edge of the view)."""
    me = obs["self"]
    reach = climb_reach(obs)
    if walkable_out(obs, reach) and not me.get("swimming"):   # on land that leads away: out
        for k in CLIMB + ("escape_target", "swim_heading"):
            memory.pop(k, None)
        return memory.get("no_path", 0) >= STUCK_AFTER
    if memory.get("climb_to") is not None:        # cutting a stair: on until up at the wall's top
        if me.get("z", 0) < memory["climb_to"] and memory.get("climb_steps", 0) < 40:
            return True
        for k in CLIMB:
            memory.pop(k, None)
    if memory.get("no_path", 0) >= STUCK_AFTER:
        return True
    grid, r = obs["grid"], len(obs["grid"]) // 2
    if me.get("swimming"):
        return not any(grid[r + q[1]][r + q[0]] in WALK_CHARS for q in reach)
    return not any(at_border(q, r) for q in reach)


def way_out(obs, memory, banned=()) -> tuple[str, dict, str, str] | None:
    """(tool, args, why, routine): routines in order bank, swim, stair; a banned one (watchdog) is skipped."""
    me, grid = obs["self"], obs["grid"]
    r = len(grid) // 2
    x, y = me["x"], me["y"]
    z = me.get("z", 0)
    if memory.get("stair") and "stair" not in banned:    # cutting a stair: the next cut, or the jump onto it
        step = memory["stair"].pop(0) if len(memory["stair"]) > 1 else memory.pop("stair")[0]
        return (*step, "stair")
    target = memory.get("escape_target")          # (routine, x, y): kept until reached, failed or banned
    if target and target[0] not in banned and abs(target[1] - x) + abs(target[2] - y) > 1:
        return "move_to", {"x": target[1], "y": target[2]}, f"getting out: on to ({target[1]},{target[2]})", target[0]
    memory.pop("escape_target", None)
    tried = memory.get("escape_avoid", set())
    reach = {q: n for q, n in climb_reach(obs).items() if (x + q[0], y + q[1]) not in tried}
    out = walkable_out(obs, reach)
    if out and "bank" not in banned:              # land reachable that goes on past the view: walk there
        _, (dx, dy) = out[0]
        memory["escape_target"] = ("bank", x + dx, y + dy)
        return "move_to", {"x": x + dx, "y": y + dy}, "getting out: to land that leads away", "bank"
    edge = [q for q in reach if at_border(q, r) and grid[r + q[1]][r + q[0]] in WATER]
    if edge and "swim" not in banned:             # swim on along the water: a heading kept, a target far off
        hd = memory.get("swim_heading")
        if hd is None or max(q[0] * hd[0] + q[1] * hd[1] for q in edge) <= 0:
            old = hd
            far = max(edge, key=lambda q: abs(q[0]) + abs(q[1]))
            hd = (far[0] > 0) - (far[0] < 0), (far[1] > 0) - (far[1] < 0)
            if old and hd[0] * old[0] + hd[1] * old[1] > 0:
                hd = (-hd[0], -hd[1])             # (the old heading ran out: the other way)
        dx, dy = max(edge, key=lambda q: (q[0] * hd[0] + q[1] * hd[1], -abs(q[0] * hd[1] - q[1] * hd[0])))
        if dx * hd[0] + dy * hd[1] > 0:
            memory["swim_heading"] = hd
            memory["escape_target"] = ("swim", x + dx, y + dy)
            return "move_to", {"x": x + dx, "y": y + dy}, "getting out: swimming on to find a low bank", "swim"
    heights = obs.get("heights")
    if not heights or "stair" in banned:
        return None
    return stair(obs, memory, reach, z)


def stair(obs, memory, reach, z):
    grid, heights = obs["grid"], obs["heights"]
    r = len(grid) // 2
    x, y = obs["self"]["x"], obs["self"]["y"]
    open_ = WATER | WALK_CHARS | {"@"}
    walls = [(level(heights[r + dy][r + dx]), d) for d, (dx, dy) in DIRS.items() if grid[r + dy][r + dx] not in open_
             or level(heights[r + dy][r + dx]) - z > 1]
    if not walls:                                 # not against a wall: go to the foot of the nearest one
        side = [(n, q) for q, n in reach.items() if q != (0, 0) and any(
            abs(q[0] + a) <= r and abs(q[1] + b) <= r and level(heights[r + q[1] + b][r + q[0] + a])
            - level(heights[r + q[1]][r + q[0]]) > 1 for a, b in DIRS.values())]
        if not side:
            return None
        _, (dx, dy) = min(side)
        return "move_to", {"x": x + dx, "y": y + dy}, "getting out: to the foot of the wall", "stair"
    top, d = min(walls)                           # cut a stair into the lowest wall: head-height block out, jump on
    if memory.get("stair_dir") in [w[1] for w in walls]:
        d = memory["stair_dir"]                   # (straight on into the same wall, a layer a time)
    memory["stair_dir"] = d
    memory.setdefault("climb_to", top)
    memory["climb_steps"] = memory.get("climb_steps", 0) + 1
    memory["stair"] = [("dig", {"dir": d, "z": z + 2}, "getting out: cutting a step into the wall"),
                       ("jump", {"dir": d}, "climbing out: onto the step")]
    return "dig", {"dir": "up"}, "getting out: room over my head for the climb", "stair"


def escape_failed(memory, tool, args):
    """A way out that did not work is not tried again (from this view). (Nothing over the head to dig: fine.)"""
    if tool == "dig" and args.get("dir") == "up":
        return
    memory.pop("stair", None)
    memory.pop("escape_target", None)
    if tool == "move_to":
        memory.setdefault("escape_avoid", set()).add((args.get("x"), args.get("y")))


def stuck_line(obs) -> str:
    me = obs["self"]
    where = "in water at the foot of a cliff" if me.get("swimming") else "in a hole or walled in"
    return (f"You are stuck: no path from here to the land ({where}). Get out first: swim along the water to a low "
            "bank, or dig the block in front at z = your z + 2 and jump onto it, a layer at a time.")
