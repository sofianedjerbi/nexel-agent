"""The `pursue` verb (#22): the goal's next step, as one game tool call - the "how" of a goal, never the "when" (the
policy chooses pursue or anything else: flee, eat, fight back, and when to come back to it). Read from the game's own
tables: what a recipe takes and where it is made (tiles.RECIPES), what a block drops (tiles.BLOCKS), the server's
own path finding (move_to).

- gather X: a pile with X on the ground (walk there, pick it up), else the nearest block in view that drops X (walk
  beside it, dig it with the right tool from the pack);
- craft X: what X still lacks, one step down, the first it can get at now (gathered where something in view gives
  it - stone from a rock, though stone is made too -, else crafted); its
  station within reach (walk to one in view; carried, put down; else make one); then the craft;
- ask, give, stay (a person): walk toward them, where they are or were last seen; in reach, the ask or the give.

None: no step to take now (an item no recipe makes and nothing in view drops, a person never seen, near enough to
stay). Only numpy-free Python: it runs in the game's Python."""

import math

from nexel_worlds.sim.bodies import in_reach, reach_of
from nexel_worlds.sim.constants import VIEW_R
from nexel_worlds.tiles import BLOCKS, RECIPES

ASK_R = 4                 # an ask is said this near the person (as a player walks up to someone to ask)
DEPTH = 4                 # a recipe chain this deep at most (pickaxe: workbench: wood)
DROPS = {}                # item: the chars of the blocks that drop it
for _b in BLOCKS:
    for _it in _b.drops:
        DROPS.setdefault(_it, set()).add(_b.char)
STATION_CH = {b.name: b.char for b in BLOCKS}


def _cells(obs, chars, blocked=()):
    """(dx, dy) of the tiles in view showing one of `chars`, nearest first, less those a walk found no way to."""
    grid, me = obs.get("grid") or [], obs["self"]
    out = [(x - VIEW_R, y - VIEW_R) for y, row in enumerate(grid) for x, ch in enumerate(row) if ch in chars
           and (me["x"] + x - VIEW_R, me["y"] + y - VIEW_R) not in blocked]
    return sorted(out, key=lambda d: math.hypot(*d))


def _walk(obs, dx, dy, ground_ch=None):
    """move_to the thing: the server's path finding goes beside what nobody can stand on (sim/movement.py)."""
    return "move_to", {"x": obs["self"]["x"] + dx, "y": obs["self"]["y"] + dy}


def gather(obs, item, ground_ch, tool_for, dirs, blocked=()):
    me = obs["self"]
    pile = next(((dx, dy) for dx, dy, items in _piles(obs) if items.get(item) and (me["x"] + dx, me["y"] + dy) not in blocked), None)
    if pile is not None:
        return ("pick_up", {}) if pile == (0, 0) else ("move_to", {"x": obs["self"]["x"] + pile[0], "y": obs["self"]["y"] + pile[1]})
    cell = next(iter(_cells(obs, DROPS.get(item, ()), blocked)), None)
    if cell is None:
        return None
    d = next((k for k, v in dirs.items() if v == cell), None)
    if d is None:
        return _walk(obs, *cell, ground_ch)
    w = tool_for(obs, d)
    return "dig", {"dir": d, **({"with": w} if w else {})}


def _piles(obs):
    return [(p["dx"], p["dy"], dict(p["items"])) for p in obs.get("items") or [] if p.get("items")]


def craft(obs, item, ground_ch, tool_for, dirs, depth=0, blocked=()):
    r = RECIPES.get(item)
    if r is None or depth > DEPTH:
        return None
    inv = obs["self"].get("inventory", {})
    lacks = [k for k, n in r.inputs.items() if inv.get(k, 0) < n]
    for lack in lacks:        # (what it can get at now - the stone in view while no tree is -: dug or picked up
        step = gather(obs, lack, ground_ch, tool_for, dirs, blocked)                  # where something here gives it,
        step = step or craft(obs, lack, ground_ch, tool_for, dirs, depth + 1, blocked)     # else made)
        if step is not None:
            return step
    if lacks:
        return None
    if r.station:
        ch = STATION_CH[r.station]
        at = next(iter(_cells(obs, {ch}, blocked)), None)
        if at is None:
            if inv.get(r.station):
                d = next((k for k, (dx, dy) in dirs.items() if obs["grid"][VIEW_R + dy][VIEW_R + dx] in ground_ch - {"~"}), None)
                return ("place", {"dir": d, "item": r.station}) if d else None
            return craft(obs, r.station, ground_ch, tool_for, dirs, depth + 1, blocked)
        if not in_reach(at[0], at[1], 0, reach_of("station")):
            return _walk(obs, *at, ground_ch)
    return "craft", {"recipe": item}


def with_person(obs, goal, social, p, wanted, handed, ground_ch):
    if p is None:
        return None
    d = math.hypot(p["dx"], p["dy"])
    if goal.kind == "ask" and p.get("seen") and d <= ASK_R:
        it = wanted(obs, goal)
        return ("say", {"text": f"could you spare some {it}?", "ask": it}) if it and (social is None or social.can_talk(obs["tick"])) else None
    if goal.kind == "give" and p.get("seen") and in_reach(p["dx"], p["dy"], 0, reach_of("give")):
        it = handed(obs, goal, social, goal.person)
        return ("give", {"target": p["id"], "item": it, "n": 1}) if it and p.get("id") is not None else None
    if goal.kind == "stay" and d <= 2:
        return None                                       # (beside them already: staying is the policy's)
    return _walk(obs, round(p["dx"]), round(p["dy"]), ground_ch) if d > 1.5 else None


def next_step(obs, goal, social, *, person, wanted, handed, tool_for, ground_ch, dirs):
    """The goal's next tool call (tool, args), or None. The helpers are rl/actions.py's own (one reading of a goal)."""
    if goal is None or goal.kind == "free":
        return None
    if goal.kind == "gather" and goal.item:
        return gather(obs, goal.item, ground_ch, tool_for, dirs, goal.blocked)
    if goal.kind == "craft" and goal.item:
        return craft(obs, goal.item, ground_ch, tool_for, dirs, blocked=goal.blocked)
    if goal.kind in ("ask", "give", "stay"):
        return with_person(obs, goal, social, person(obs, goal), wanted, handed, ground_ch)
    return None
