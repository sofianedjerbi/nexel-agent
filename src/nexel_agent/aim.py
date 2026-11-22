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


BACKOFF_S = 1.0                                  # an action that just failed is not sent again this soon


def just_failed(memory, tool, args, now) -> bool:
    last = memory.get("last_fail")
    return bool(last) and last[:2] == (tool, args) and now - last[2] < BACKOFF_S


def note_outcome(memory, failed, obs, item, reply, t1):
    """What one result teaches the body: failures by tile, the last failure (backoff), no_path in a row."""
    from .escape import escape_failed
    ok, err = bool(reply.get("ok")), reply.get("error")
    if not ok and err not in ("interrupted", "rate_limited", "hands_full"):
        f = failed.setdefault(fail_key(obs, item), [0, ""])
        f[0], f[1] = f[0] + 1, str(err)
        memory["last_fail"] = (item.tool, item.args, t1)
    if item.tool in ("move_to", "step", "walk", "jump") or err == "no_path":
        memory["no_path"] = memory.get("no_path", 0) + 1 if err == "no_path" else 0
    if item.source == "escape" and not ok:
        escape_failed(memory, item.tool, item.args)


def fail_key(obs, it) -> tuple:
    """What a failure is remembered by: the tool, what it was used with, and the tile it was aimed at."""
    me = obs["self"]
    d = DIRS.get(it.args.get("dir"), (0, 0))
    tile = (it.args["x"], it.args["y"]) if "x" in it.args and "y" in it.args else (me["x"] + d[0], me["y"] + d[1])
    return it.tool, it.args.get("with") or it.args.get("item") or it.args.get("recipe"), tile
