"""Scripted policy: used by M1 bots and as the thinker's fallback while the model is busy.

It reads only the observation dict the API returns, so it is a fair baseline for the LLM agents.
"""

import random

from .tiles import DIRS, FOOD

# grid char -> what we can harvest there
TARGETS = {"T": "scrap_heap", "b": "lichen", "w": "ripe_bed", "H": "conduit", "Q": "terminal"}
SOLID_CHARS = set("TMHQ*#%&C")
WALKABLE = set(".,:_'o+?!\"wbt$")


def find(obs, chars, avoid=()):
    """Nearest grid cell (dx, dy) whose char is in `chars`, by Manhattan distance."""
    grid = obs["grid"]
    r = len(grid) // 2
    x0, y0 = obs["self"]["x"] - r, obs["self"]["y"] - r
    best = None
    for j, row in enumerate(grid):
        for i, ch in enumerate(row):
            if ch in chars and (x0 + i, y0 + j) not in avoid:
                d = abs(i - r) + abs(j - r)
                if d and (best is None or d < best[0]):
                    best = (d, i - r, j - r)
    return best


def free_dirs(obs):
    """Directions whose adjacent tile is empty floor (a block can be placed there)."""
    grid, r = obs["grid"], len(obs["grid"]) // 2
    return [d for d, (dx, dy) in DIRS.items() if grid[r + dy][r + dx] in ".,:_'t"]


def adjacent_dir(dx, dy):
    for name, (ddx, ddy) in DIRS.items():
        if (ddx, ddy) == (dx, dy):
            return name
    return None


def feedback(memory: dict, reply: dict, tick: int):
    """Update policy memory with an action outcome: remember unreachable targets, count failures."""
    if reply.get("ok"):
        memory["fails"] = 0
        return
    memory["fails"] = memory.get("fails", 0) + 1
    target = memory.get("target")
    if target is not None:
        memory.setdefault("avoid", {})[target] = tick + 1500


def scripted(obs: dict, rng: random.Random, persona: str = "", memory: dict | None = None) -> tuple[str, dict, str]:
    """Return (tool, args, reason). `memory` is a dict the caller keeps between calls."""
    memory = {} if memory is None else memory
    memory["target"] = None
    avoid = {p for p, until in memory.get("avoid", {}).items() if until > obs["tick"]}
    me = obs["self"]
    inv = me["inventory"]
    x, y = me["x"], me["y"]

    # survival
    if me["hunger"] < 60:
        food = max((f for f in FOOD if inv.get(f)), key=lambda f: FOOD[f], default=None)
        if food:
            return "eat", {"item": food}, f"hungry ({me['hunger']}), eating {food}"
    if me["energy"] < 15:
        return "rest", {}, "exhausted"
    for e in obs["entities"]:
        if e["kind"] == "sentinel" and abs(e["dx"]) + abs(e["dy"]) <= 4:
            return "move_to", {"x": x - 3 * (e["dx"] > 0) + 3 * (e["dx"] < 0),
                               "y": y - 3 * (e["dy"] > 0) + 3 * (e["dy"] < 0)}, "fleeing a sentinel"
    if any(n for n in obs["items"] if n["dx"] == 0 and n["dy"] == 0):
        return "pick_up", {}, "items underfoot"

    # crafting ladder
    if inv.get("scrap", 0) >= 4 and not inv.get("bench") and not find(obs, "&"):
        return "craft", {"recipe": "bench"}, "need a bench"
    free = free_dirs(obs)
    if inv.get("bench") and free:
        return "place", {"dir": rng.choice(free), "item": "bench"}, "placing bench"
    near_bench = (b := find(obs, "&")) and b[0] <= 2
    if near_bench and inv.get("scrap", 0) >= 2 and inv.get("alloy", 0) >= 2 and inv.get("cable") and not inv.get("cutter"):
        return "craft", {"recipe": "cutter"}, "making a cutter"
    if "builder" in persona and inv.get("scrap", 0) >= 6:
        return "craft", {"recipe": "scrap_wall", "n": 2}, "builder: making walls"
    if inv.get("scrap_wall") and len(free) >= 2:
        return "place", {"dir": rng.choice(free), "item": "scrap_wall"}, "building a wall"

    # stuck (walled in, or everything nearby unreachable): dig out toward a random side
    if memory.get("fails", 0) >= 4:
        memory["fails"] = 0
        d = rng.choice(list(DIRS))
        return "dig", {"dir": d}, "stuck: digging out"

    # harvesting
    wanted = "bT" if me["hunger"] < 80 else "TbHQM"
    t = find(obs, wanted, avoid) or find(obs, "TbwHQ", avoid)
    if t:
        d, dx, dy = t
        if d == 1:
            return "dig", {"dir": adjacent_dir(dx, dy)}, f"harvesting {obs['legend'].get(obs['grid'][len(obs['grid']) // 2 + dy][len(obs['grid']) // 2 + dx])}"
        # stand next to it (not on it: non-solid targets like lichen can't be dug from on top),
        # on a walkable neighbour, nearest to us
        grid, r = obs["grid"], len(obs["grid"]) // 2
        spots = []
        for ddx, ddy in DIRS.values():
            sx, sy = dx + ddx, dy + ddy
            if abs(sx) <= r and abs(sy) <= r and (sx, sy) != (0, 0) and grid[r + sy][r + sx] in WALKABLE:
                spots.append((abs(sx) + abs(sy), sx, sy))
        if spots:
            _, sx, sy = min(spots)
            memory["target"] = (x + dx, y + dy)
            return "move_to", {"x": x + sx, "y": y + sy}, "walking next to a resource"

    # explore
    dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    return "move_to", {"x": x + dx * rng.randint(6, 14), "y": y + dy * rng.randint(6, 14)}, "exploring"
