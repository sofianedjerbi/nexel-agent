"""Knowing you are stuck (gatsu0, 2026-11-22: swimming in a river at the foot of cliffs 10 layers high, every move_to
to the land failed): no walk gets anywhere from here. The model is told so, and how a body gets out (swim along to a
low bank, or cut a step and jump); what to do is its own choice: nothing here moves the body.
"""

from .policy import WALK_CHARS
from .tiles import DIRS

WATER = set("~(=)")
STUCK_AFTER = 3                                  # no_path in a row before the body calls itself stuck


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
        return memory.get("no_path", 0) >= STUCK_AFTER
    if memory.get("no_path", 0) >= STUCK_AFTER:
        return True
    grid, r = obs["grid"], len(obs["grid"]) // 2
    if me.get("swimming"):
        return not any(grid[r + q[1]][r + q[0]] in WALK_CHARS for q in reach)
    return not any(at_border(q, r) for q in reach)


def stuck_line(obs) -> str:
    me = obs["self"]
    where = "in water at the foot of a cliff" if me.get("swimming") else "in a hole or walled in"
    return (f"No path leads from here to the land ({where}). Bodies get out of such places by swimming along the "
            "water to a low bank, or by digging the block in front at z = your z + 2 and jumping onto it, a layer at a time.")
