"""What the real-time player does while the model is still thinking: the current goal's obvious next step.

Only gathering, walking and fleeing: never crafting, placing, giving or fighting, so it never spends what the
model's plan needs or does something the plan would not. An unknown goal gathers the nearest useful thing.
"""

from .policy import approach, find, reachable

GOALS = (                                    # (words in the goal, chars to dig, tool that helps)
    (("wood", "tree", "log", "plank", "workbench", "chest", "bed", "fence", "door", "boat", "house", "shelter", "build"),
     "Tt", "axe"),
    (("stone", "rock", "pickaxe", "wall", "path"), "M", "pickaxe"),
    (("ore", "iron", "ingot", "sword", "armor", "gold"), "*M", "pickaxe"),
    (("food", "berr", "eat", "hungry", "seed", "farm"), "bQF", None),
    (("fiber", "rope", "string", "torch", "cloth"), "vbH", "axe"),
    (("mushroom",), "Q", None),
)
DANGER = ("tusker", "pincer", "strawman", "wyrm", "mossback")
NEEDS = {"*": "pickaxe"}                    # cannot be dug without it
TOOL_FOR = {"T": "axe", "H": "axe", "M": "pickaxe", "*": "pickaxe"}


def filler(obs, goal: str | None, memory: dict, rng) -> tuple[str, dict, str] | None:
    """(tool, args, why) in the spirit of `goal`, or None when nothing sensible is in view."""
    me = obs["self"]
    x, y = me["x"], me["y"]
    inv = me["inventory"]
    memory["target"] = None
    reach = memory["_reach"] = reachable(obs)
    for e in obs["entities"]:
        if e["kind"] in DANGER and abs(e["dx"]) + abs(e["dy"]) <= 4:
            far = max(((abs(q[0] - e["dx"]) + abs(q[1] - e["dy"]), q) for q, n in reach.items() if 2 <= n <= 5),
                      default=None)
            if far:
                return "move_to", {"x": x + far[1][0], "y": y + far[1][1]}, f"stepping away from a {e['kind']}"
    if any(i["dx"] == 0 and i["dy"] == 0 for i in obs["items"]):
        return "pick_up", {}, "picking up what is here"
    avoid = {p for p, until in memory.get("avoid", {}).items() if until > obs["tick"]}
    if memory.get("fails", 0) >= 3:              # what it tried here keeps failing: go somewhere else
        memory["fails"] = 0
        return explore(obs, reach, avoid, memory, rng)
    g = (goal or "").lower()
    if any(w in g for w in ("explore", "look", "find", "search", "wander", "scout")):
        return explore(obs, reach, avoid, memory, rng)
    if any(w in g for w in ("home", "sleep", "bed")) and me.get("spawn"):
        sx, sy = me["spawn"]
        if abs(sx - x) + abs(sy - y) > 3:
            return "move_to", {"x": sx, "y": sy + 1}, "heading home"
    for words, chars, _ in GOALS:
        if any(w in g for w in words):
            act = gather(obs, chars, inv, avoid, memory, f"{goal}: gathering")
            if act:
                return act
    for e in obs["entities"]:                # a goal about someone: go near them
        if e["kind"] == "agent" and e["name"].lower() in g and abs(e["dx"]) + abs(e["dy"]) > 2:
            return "move_to", {"x": x + e["dx"] - (e["dx"] > 0) + (e["dx"] < 0), "y": y + e["dy"]}, f"going to {e['name']}"
    act = gather(obs, "TtbvQFM", inv, avoid, memory, "gathering while thinking")
    return act or explore(obs, reach, avoid, memory, rng)


def gather(obs, chars, inv, avoid, memory, why):
    chars = "".join(c for c in chars if c not in NEEDS or inv.get(NEEDS[c]))
    t = find(obs, chars, avoid, max_d=8) if chars else None
    if not t:
        return None
    me = obs["self"]
    tx, ty = me["x"] + t[1], me["y"] + t[2]
    d, move = approach(obs, memory, tx, ty, why)
    memory["target"] = (tx, ty)                  # a failure here puts it on the avoid list (policy.feedback)
    if move:
        return move
    ch = obs["grid"][len(obs["grid"]) // 2 + t[2]][len(obs["grid"]) // 2 + t[1]]
    tool = TOOL_FOR.get(ch)
    return "dig", ({"dir": d, "with": tool} if tool and inv.get(tool) else {"dir": d}), why


def explore(obs, reach, avoid, memory, rng):
    me = obs["self"]
    r = len(obs["grid"]) // 2
    edge = [q for q, n in reach.items() if 4 <= n <= 8 and (me["x"] + q[0], me["y"] + q[1]) not in avoid]
    if not edge:
        edge = [q for q, n in reach.items() if n >= 1]
    if not edge:
        return None
    heading = memory.get("heading")              # keep a heading for a while: no back-and-forth
    if heading:
        edge.sort(key=lambda q: -(q[0] * heading[0] + q[1] * heading[1]))
        dx, dy = edge[rng.randrange(min(3, len(edge)))]
    else:
        dx, dy = rng.choice(edge)
        memory["heading"] = (dx, dy)
    if rng.random() < 0.1:
        memory["heading"] = None
    memory["target"] = (me["x"] + dx, me["y"] + dy)
    return "move_to", {"x": me["x"] + dx, "y": me["y"] + dy}, f"exploring (view {r})"
