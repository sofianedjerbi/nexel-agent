"""Scripted policy: used by bots and as the thinker's fallback while the model is busy.

It reads only the observation dict the API returns, so it is a fair baseline for learned or LLM agents.
Needs are tried in priority order: survive (eat, flee what bites, sleep), set up a home (workbench, bed, chest),
tools, farming next to home, persona extras, then gathering what is still missing, then exploring.
"""

import random

from .tiles import BLOCKS, DIRS, FOOD, GROUNDS

WALK_CHARS = {g.char for g in GROUNDS if g.walkable} | {b.char for b in BLOCKS if not b.solid and b.name != "none"}
OPEN_GROUND = {"grass": ".", "dirt": ","}
FREE_CHARS = {".", ",", "'", "t", "f"}     # a block can be placed here (stumps and flowers are cleared)
STOCK = {"wood": 30, "stone": 20, "fiber": 12, "berries": 10}   # beyond this, stop gathering, store the rest
KEEP = {"wood": 12, "stone": 8, "fiber": 6}                     # what to carry after storing


def find(obs, chars, avoid=(), max_d=99):
    """Nearest grid cell (d, dx, dy) whose char is in `chars`, by Manhattan distance."""
    grid = obs["grid"]
    r = len(grid) // 2
    x0, y0 = obs["self"]["x"] - r, obs["self"]["y"] - r
    best = None
    for j, row in enumerate(grid):
        for i, ch in enumerate(row):
            if ch in chars and (x0 + i, y0 + j) not in avoid:
                d = abs(i - r) + abs(j - r)
                if d and d <= max_d and (best is None or d < best[0]):
                    best = (d, i - r, j - r)
    return best


def char_at(obs, x, y):
    """Grid char at absolute (x, y), or None outside the view."""
    grid, r = obs["grid"], len(obs["grid"]) // 2
    dx, dy = x - obs["self"]["x"], y - obs["self"]["y"]
    if abs(dx) > r or abs(dy) > r:
        return None
    return grid[r + dy][r + dx]


def reachable(obs) -> dict:
    """Tiles in view we can walk to (relative (dx, dy) -> steps), by flood fill from where we stand."""
    grid, r = obs["grid"], len(obs["grid"]) // 2
    seen, frontier = {(0, 0): 0}, [(0, 0)]
    while frontier:
        nxt = []
        for cx, cy in frontier:
            for ddx, ddy in DIRS.values():
                q = (cx + ddx, cy + ddy)
                if q not in seen and abs(q[0]) <= r and abs(q[1]) <= r and grid[r + q[1]][r + q[0]] in WALK_CHARS:
                    seen[q] = seen[(cx, cy)] + 1
                    nxt.append(q)
        frontier = nxt
    return seen


def free_dirs(obs):
    """Directions whose adjacent tile is empty ground (a block can be placed there)."""
    grid, r = obs["grid"], len(obs["grid"]) // 2
    return [d for d, (dx, dy) in DIRS.items() if grid[r + dy][r + dx] in FREE_CHARS]


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
        memory.setdefault("avoid", {})[tuple(target)] = tick + 1500


def approach(obs, memory, tx, ty, why):
    """Act on the absolute tile (tx, ty) if adjacent, otherwise walk to its nearest open side.
    Returns (dir or None, move action or None)."""
    me = obs["self"]
    x, y = me["x"], me["y"]
    dx, dy = tx - x, ty - y
    if abs(dx) + abs(dy) == 1:
        return adjacent_dir(dx, dy), None
    reach = memory.get("_reach") or reachable(obs)
    spots = []
    for ddx, ddy in DIRS.values():
        sx, sy = tx + ddx, ty + ddy
        if (sx - x, sy - y) in reach and (sx, sy) != (x, y):
            spots.append((reach[(sx - x, sy - y)], sx, sy))
    memory["target"] = (tx, ty)
    if not spots and char_at(obs, tx, ty) is not None:
        return None, ("move_to", {"x": tx, "y": ty}, why + " (no way there)")   # fails, and feedback avoids it
    if not spots:
        return None, ("move_to", {"x": tx, "y": ty + 1}, why)   # out of view: head that way
    _, sx, sy = min(spots)
    return None, ("move_to", {"x": sx, "y": sy}, why)


def scripted(obs: dict, rng: random.Random, persona: str = "", memory: dict | None = None) -> tuple[str, dict, str]:
    """Return (tool, args, reason). `memory` is a dict the caller keeps between calls."""
    memory = {} if memory is None else memory
    memory["target"] = None
    reach = memory["_reach"] = reachable(obs)
    avoid = {p for p, until in memory.get("avoid", {}).items() if until > obs["tick"]}
    me = obs["self"]
    inv = me["inventory"]
    x, y = me["x"], me["y"]
    night = "night" in obs.get("time", "")
    has = lambda k, n=1: inv.get(k, 0) >= n   # noqa: E731

    def act_on(tx, ty, tool, args, why):
        d, move = approach(obs, memory, tx, ty, why)
        return ("move_to", *move[1:]) if move else (tool, {**args, "dir": d}, why)

    # ---------------------------------------------------------------- survive
    if me["hunger"] < 12:                  # (food is 0 to 20)
        food = max((f for f in FOOD if inv.get(f) and f not in ("wheat",)), key=lambda f: FOOD[f], default=None)
        food = food or ("wheat" if has("wheat") else None)
        if food:
            return "eat", {"item": food}, f"hungry ({me['hunger']}), eating {food}"
    for e in obs["entities"]:
        if e["kind"] in ("tusker", "pincer", "strawman", "wyrm", "mossback") and abs(e["dx"]) + abs(e["dy"]) <= 4:
            # run to the reachable tile 3-6 steps away that is farthest from it
            far = max(((abs(q[0] - e["dx"]) + abs(q[1] - e["dy"]), q) for q, n in reach.items() if 3 <= n <= 6),
                      default=None)
            if far:
                return "move_to", {"x": x + far[1][0], "y": y + far[1][1]}, f"fleeing a {e['kind']}"
    if any(n for n in obs["items"] if n["dx"] == 0 and n["dy"] == 0):
        return "pick_up", {}, "items underfoot"
    bed = me.get("spawn") or memory.get("bed")
    if bed and abs(bed[0] - x) + abs(bed[1] - y) > 40:
        bed = None                         # an old bed far away: settle here instead
    if night and bed:                      # nights at home, by the bed (it's where you wake up after dying)
        if char_at(obs, *bed) == "B" or char_at(obs, *bed) is None:
            return act_on(bed[0], bed[1], "sleep", {}, "night: going to bed")
        memory.pop("bed", None)           # the bed is gone

    # stuck (walled in, or everything nearby unreachable): dig out toward a random side
    if memory.get("fails", 0) >= 4:
        memory["fails"] = 0
        grid, r = obs["grid"], len(obs["grid"]) // 2
        blocked = [d for d, (dx, dy) in DIRS.items() if grid[r + dy][r + dx] not in WALK_CHARS | {"~", "="}]
        if blocked:
            return "dig", {"dir": rng.choice(blocked)}, "stuck: digging out"
        memory["home"] = memory["bed"] = None   # can't get anywhere from here: start over

    # ---------------------------------------------------------------- home: workbench, bed, chest
    free = free_dirs(obs)
    home = memory.get("home")
    if home and char_at(obs, *home) not in (None, "&"):
        home = memory["home"] = None      # someone took our workbench
    if home and abs(home[0] - x) + abs(home[1] - y) > 40:
        home = memory["home"] = None
    if not home:
        b = find(obs, "&", max_d=6)
        if b:
            home = memory["home"] = (x + b[1], y + b[2])
    if not home:
        if has("workbench") and free:
            d = rng.choice(free)
            memory["home"] = (x + DIRS[d][0], y + DIRS[d][1])
            return "place", {"dir": d, "item": "workbench"}, "settling: placing a workbench"
        if has("wood", 4):
            return "craft", {"recipe": "workbench"}, "settling: making a workbench"
    at_home = home and abs(home[0] - x) + abs(home[1] - y) <= 2

    def at_bench(recipe, why):
        """Craft at the home workbench, walking back to it first."""
        if at_home:
            return "craft", {"recipe": recipe}, why
        d, move = approach(obs, memory, home[0], home[1], f"{why} (going home)")
        return move or ("craft", {"recipe": recipe}, why)

    if home:
        if not has("axe") and has("wood", 2) and has("stone", 2) and has("fiber"):
            return at_bench("axe", "making an axe")
        if not has("pickaxe") and has("axe") and has("wood", 2) and has("stone", 3) and has("fiber"):
            return at_bench("pickaxe", "making a pickaxe")
        if not has("hoe") and has("seed") and has("wood", 2) and has("stone", 1):
            return at_bench("hoe", "making a hoe to farm")
        if not bed and not has("bed") and has("wood", 6) and has("fiber", 3):
            return at_bench("bed", "making a bed")
        if has("bed") and not bed and at_home and free:
            d = rng.choice(free)
            memory["bed"] = (x + DIRS[d][0], y + DIRS[d][1])
            return "place", {"dir": d, "item": "bed"}, "placing my bed by the workbench"
        chest = memory.get("chest")
        if chest and char_at(obs, *chest) not in (None, "C"):
            chest = memory["chest"] = None
        if not chest:
            c = find(obs, "C", max_d=4) if at_home else None
            if c:
                chest = memory["chest"] = (x + c[1], y + c[2])
        heavy = [k for k in KEEP if inv.get(k, 0) > STOCK.get(k, 99)]
        if heavy and not chest:
            if has("chest") and at_home and free:
                d = rng.choice(free)
                memory["chest"] = (x + DIRS[d][0], y + DIRS[d][1])
                return "place", {"dir": d, "item": "chest"}, "placing a chest for my stock"
            if not has("chest") and has("wood", 6):
                return at_bench("chest", "too much to carry: making a chest")
        if heavy and chest:
            k = heavy[0]
            return act_on(chest[0], chest[1], "store", {"item": k, "n": inv[k] - KEEP[k]}, f"storing {k}")

        # ------------------------------------------------------------ farm next to home
        ripe = find(obs, "w", avoid, max_d=12)
        if ripe:
            return act_on(x + ripe[1], y + ripe[2], "dig", {}, "harvesting a ripe crop")
        if has("seed") and at_home:
            spot = find(obs, "o", avoid, max_d=3)
            if spot:
                return act_on(x + spot[1], y + spot[2], "place", {"item": "seed"}, "planting a seed")
            spot = find(obs, ".", avoid, max_d=3) if has("hoe") else None
            if spot:
                return act_on(x + spot[1], y + spot[2], "dig", {"with": "hoe"}, "tilling soil for seeds")

        # ------------------------------------------------------------ persona extras
        if "builder" in persona and at_home:
            if has("fence"):
                if free:
                    return "place", {"dir": rng.choice(free), "item": "fence"}, "builder: fencing the yard"
            elif has("wood", 10):
                return "craft", {"recipe": "fence"}, "builder: making a fence"
        if "explorer" in persona and has("wood", 10) and not has("boat") and has("rope"):
            return at_bench("boat", "explorer: building a boat")
        if "explorer" in persona and has("fiber", 3) and not has("rope"):
            return "craft", {"recipe": "rope"}, "making rope"

    # ---------------------------------------------------------------- gather what is missing
    need = []
    if me["hunger"] < 16 or not any(inv.get(f) for f in FOOD):
        need += ["bFQ"]                    # berries, fruit, mushrooms
    if inv.get("wood", 0) < STOCK["wood"]:
        need += ["Tt"]
    if inv.get("fiber", 0) < STOCK["fiber"]:
        need += ["vbH"]
    if inv.get("stone", 0) < STOCK["stone"]:
        need += ["M*" if has("pickaxe") else "M"]
    for chars in need:
        t = find(obs, chars, avoid)
        if t:
            ch = char_at(obs, x + t[1], y + t[2])
            tool = {"T": "axe", "H": "axe", "M": "pickaxe", "*": "pickaxe"}.get(ch)   # the right tool in hand
            return act_on(x + t[1], y + t[2], "dig", {"with": tool} if tool and has(tool) else {},
                          f"gathering: {obs['legend'].get(ch, ch)}")

    # ---------------------------------------------------------------- explore, but come home for the night
    if home and (night or rng.random() < 0.3) and abs(home[0] - x) + abs(home[1] - y) > 6:
        return "move_to", {"x": home[0], "y": home[1] + 1}, "heading home"
    # explore: a random walkable tile near the edge of the view (a target in the water has no path)
    grid, r = obs["grid"], len(obs["grid"]) // 2
    edge = [q for q, n in reach.items() if max(abs(q[0]), abs(q[1])) >= r - 3 and (x + q[0], y + q[1]) not in avoid]
    if edge:
        dx, dy = rng.choice(edge)
        memory["target"] = (x + dx, y + dy)
        return "move_to", {"x": x + dx, "y": y + dy}, "exploring"
    dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    return "move_to", {"x": x + dx * 8, "y": y + dy * 8}, "exploring"
