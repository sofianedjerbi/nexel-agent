"""The `pursue` verb (#22): the goal's next step, as one game tool call - the "how" of a goal, never the "when" (the
policy chooses pursue or anything else: flee, eat, fight back, and when to come back to it). Read from the game's own
tables: what a recipe takes and where it is made (tiles.RECIPES), what a block drops (tiles.BLOCKS), the server's
own path finding (move_to).

- gather X: a pile with X on the ground (walk there, pick it up), else the nearest block in view that drops X, or that
  picked leaves one that does (tiles.PICKED) (walk beside it, dig it with the right tool from the pack);
- craft X: what X still lacks, one step down, the first it can get at now (gathered where something in view gives
  it - stone from a rock, though stone is made too -, else crafted); its
  station within reach (walk to one in view; carried, put down; else make one); then the craft;
- ask, give, stay (a person): walk toward them, where they are or were last seen; in reach, the ask or the give.

None: no step to take now (an item no recipe makes and nothing in view drops, a person never seen, near enough to
stay). Only numpy-free Python: it runs in the game's Python."""

import math

from nexel_worlds.sim.bodies import in_reach, reach_of
from nexel_worlds.sim.constants import VIEW_R
from nexel_worlds.sim.observe import WATER
from nexel_worlds.tiles import BLOCKS, PICKED, RECIPES

ASK_R = 4                 # an ask is said this near the person (as a player walks up to someone to ask)
DEPTH = 4                 # a recipe chain this deep at most (pickaxe: workbench: wood)
DROPS = {}                # item: the chars of the blocks that drop it, or that picked leave one that does (a berry
_BY_NAME = {b.name: b for b in BLOCKS}        # bush picked is a bare bush, dug for fibre: the most fibre away from swamps)
for _b in BLOCKS:
    for _it in {**_b.drops, **(_BY_NAME[PICKED[_b.name][0]].drops if _b.name in PICKED else {})}:
        DROPS.setdefault(_it, set()).add(_b.char)
STATION_CH = {b.name: b.char for b in BLOCKS}
NEEDS = {b.char: b.needs_tool for b in BLOCKS if b.needs_tool}     # (ore: no stone from it by hand)


LOOK = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
LOOK_TICKS = 600                  # (a way kept for this long: far enough to see new land, then the next)
LOOK_R = 200                      # (a search from a home stays this near it: rl/build.py gives a home up past FAR)


def _rim(hs, dx, dy) -> bool:
    """Is the tile (dx, dy) from the centre on a hill's rim - a lower tile beside it, diagonals too? The game keeps
    nothing there ("cliff_edge": the cross on seed 109 put its bench down there 199 times)."""
    from .actions import LEVEL
    lv = LEVEL.get(hs[VIEW_R + dy][VIEW_R + dx])
    return lv is not None and any(LEVEL.get(hs[VIEW_R + dy + j][VIEW_R + dx + i], lv) < lv
                                  for i in (-1, 0, 1) for j in (-1, 0, 1) if 0 <= VIEW_R + dy + j < len(hs)
                                  and 0 <= VIEW_R + dx + i < len(hs[0]))


def _cells(obs, chars, blocked=()):
    """(dx, dy) of the tiles in view showing one of `chars`, nearest first, less those a walk found no way to."""
    grid, me = obs.get("grid") or [], obs["self"]
    out = [(x - VIEW_R, y - VIEW_R) for y, row in enumerate(grid) for x, ch in enumerate(row) if ch in chars
           and (me["x"] + x - VIEW_R, me["y"] + y - VIEW_R) not in blocked]
    return sorted(out, key=lambda d: math.hypot(*d))


def _walk(obs, dx, dy, ground_ch=None):
    """move_to the thing: the server's path finding goes beside what nobody can stand on (sim/movement.py)."""
    return "move_to", {"x": obs["self"]["x"] + dx, "y": obs["self"]["y"] + dy}


def gather(obs, item, ground_ch, tool_for, dirs, blocked=(), keep=(), look=False):
    """The next step to get `item`: a pile picked up, else a block that gives it dug (none of `keep`: the home's)."""
    me = obs["self"]
    pile = next(((dx, dy) for dx, dy, items in _piles(obs) if items.get(item) and (me["x"] + dx, me["y"] + dy) not in blocked), None)
    if pile is not None:
        return ("pick_up", {}) if pile == (0, 0) else ("move_to", {"x": obs["self"]["x"] + pile[0], "y": obs["self"]["y"] + pile[1]})
    inv = me.get("inventory", {})
    cell = next(iter(_cells(obs, {c for c in DROPS.get(item, ()) if c not in NEEDS or inv.get(NEEDS[c])}, set(blocked) | set(keep))), None)
    if cell is None:                     # (none in view: for a project, a view's walk to look further, the way turning
        if not look or not DROPS.get(item):          # as time goes; else finding one is the policy's, as it learned)
            return None
        k = (obs.get("tick", 0) // LOOK_TICKS) % len(LOOK)          # (the way of the time, else the next not shut)
        dx, dy = next((w for w in LOOK[k:] + LOOK[:k] if (me["x"] + w[0] * (VIEW_R - 1), me["y"] + w[1] * (VIEW_R - 1))
                       not in blocked), LOOK[k])
        if keep:                         # (a home: no further than LOOK_R from it - seed 102: 230 tiles off for stone,
            ax, ay = sum(x for x, _ in keep) / len(keep), sum(y for _, y in keep) / len(keep)    # the home given up)
            far = math.hypot(ax - me["x"], ay - me["y"])
            if far > LOOK_R:
                k = (VIEW_R - 1) / far
                return _walk(obs, round((ax - me["x"]) * k), round((ay - me["y"]) * k), ground_ch)
        return _walk(obs, dx * (VIEW_R - 1), dy * (VIEW_R - 1), ground_ch)
    d = next((k for k, v in dirs.items() if v == cell), None)
    if d is None:
        return _walk(obs, *cell, ground_ch)
    w = tool_for(obs, d)
    return "dig", {"dir": d, **({"with": w} if w else {})}


def _piles(obs):
    return [(p["dx"], p["dy"], dict(p["items"])) for p in obs.get("items") or [] if p.get("items")]


def craft(obs, item, ground_ch, tool_for, dirs, depth=0, blocked=(), keep=(), look=False):
    r = RECIPES.get(item)
    if r is None or depth > DEPTH:
        return None
    inv = obs["self"].get("inventory", {})
    lacks = [k for k, n in r.inputs.items() if inv.get(k, 0) < n]
    for lack in lacks:        # (what it can get at now - the stone in view while no tree is -: dug or picked up
        step = gather(obs, lack, ground_ch, tool_for, dirs, blocked, keep, look)      # where something here gives it,
        step = step or craft(obs, lack, ground_ch, tool_for, dirs, depth + 1, blocked, keep, look)   # else made)
        if step is not None:
            return step
    if lacks:
        return None
    if r.station and r.station not in (obs["self"].get("stations") or ()):  # (the game says which are in reach)
        ch = STATION_CH[r.station]
        at = next(iter(_cells(obs, {ch}, blocked)), None)
        if at is None:
            if inv.get(r.station):
                me = obs["self"]                     # (not on the home's own tiles: a bench put on a wall's tile
                hs = obs.get("heights")              # (and on its own level: a bench put on a slope is refused,
                free = [k for k, (dx, dy) in dirs.items() if obs["grid"][VIEW_R + dy][VIEW_R + dx] in ground_ch - {WATER}]
                level = lambda dx, dy: not hs or hs[VIEW_R + dy][VIEW_R + dx] == hs[VIEW_R][VIEW_R] and not _rim(hs, dx, dy)  # noqa: E731
                d = next((k for k in free if (me["x"] + dirs[k][0], me["y"] + dirs[k][1]) not in keep   # 580 times over,
                          and level(*dirs[k])), None) if len(free) > 1 else None       # scenario craft axe; never on its
                if d is None and (keep or free):     # last way out - house, seed 107: put down, dug back up, 611 times)
                    #                                  (every tile round it the home's or a slope: a step off the
                    off = next(((dx, dy) for dx, dy in _cells(obs, ground_ch - {WATER}, blocked)          # site first -
                                if (dx, dy) != (0, 0) and (me["x"] + dx, me["y"] + dy) not in keep), None)  # the round
                    return _walk(obs, *off, ground_ch) if off else None                     # home, 80 walls left)
                return ("place", {"dir": d, "item": r.station}) if d else None
            return craft(obs, r.station, ground_ch, tool_for, dirs, depth + 1, blocked, keep, look)
        return _walk(obs, *at, ground_ch)                   # (in view, not in reach: walk to it)
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
    from .build import own
    from .site import no_way
    blocked = set(goal.blocked) | no_way(social)
    if goal.kind == "gather" and goal.item:
        return gather(obs, goal.item, ground_ch, tool_for, dirs, blocked, own(social), getattr(goal, "look", False))
    if goal.kind == "craft" and goal.item:
        return craft(obs, goal.item, ground_ch, tool_for, dirs, blocked=blocked, keep=own(social), look=getattr(goal, "look", False))
    if goal.kind in ("ask", "give", "stay"):
        return with_person(obs, goal, social, person(obs, goal), wanted, handed, ground_ch)
    return None
