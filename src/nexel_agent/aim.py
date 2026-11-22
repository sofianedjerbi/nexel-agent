"""Checks the real-time body makes on a planned command just before sending it (the plan was made a moment ago,
on an older observation): is there still something where it digs, which way is the thing it walked to, and
what a failure is remembered by."""

from .policy import DIRS
from .tiles import FOOD

BARE = set(".,o_'nu-:")                          # plain ground: nothing to harvest
HUNGRY = 8                                       # eat at this food level or lower, when carrying food
DANGER = ("tusker", "pincer", "strawman", "wyrm", "mossback")


def reflex(obs):
    """What no plan waits for: eat when hungry (with food), step away from what bites when hurt. (tool, args, why)."""
    me = obs["self"]
    inv = me["inventory"]
    if me["hunger"] <= HUNGRY and me.get("action") != "eat":
        food = max((f for f in FOOD if inv.get(f)), key=lambda f: FOOD[f], default=None)
        if food:
            return "eat", {"item": food}, f"(reflex) hungry ({me['hunger']}/20), eating {food}"
    for e in obs["entities"]:
        if e["kind"] in DANGER and abs(e["dx"]) + abs(e["dy"]) <= 2 and me["hp"] <= 12:
            return "move_to", {"x": me["x"] - 4 * (e["dx"] > 0) + 4 * (e["dx"] <= 0),
                               "y": me["y"] - 4 * (e["dy"] > 0) + 4 * (e["dy"] <= 0)}, "(reflex) fleeing"
    return None


def gone(obs, it) -> bool:
    """A planned dig at bare ground: what the plan saw there was taken meanwhile."""
    if it.tool != "dig" or it.args.get("dir") not in DIRS or "z" in it.args or it.args.get("with") == "hoe":
        return False
    dx, dy = DIRS[it.args["dir"]]
    r = len(obs["grid"]) // 2
    return obs["grid"][r + dy][r + dx] in BARE


def facing(obs, target) -> str | None:
    """The dir from where we stand to an adjacent target tile, if it is adjacent and not bare ground."""
    me = obs["self"]
    dx, dy = target[0] - me["x"], target[1] - me["y"]
    r = len(obs["grid"]) // 2
    for d, v in DIRS.items():
        if v == (dx, dy) and obs["grid"][r + dy][r + dx] not in BARE:
            return d
    return None


def fail_key(obs, it) -> tuple:
    """What a failure is remembered by: the tool, what it was used with, and the tile it was aimed at."""
    me = obs["self"]
    d = DIRS.get(it.args.get("dir"), (0, 0))
    tile = (it.args["x"], it.args["y"]) if "x" in it.args and "y" in it.args else (me["x"] + d[0], me["y"] + d[1])
    return it.tool, it.args.get("with") or it.args.get("item") or it.args.get("recipe"), tile
