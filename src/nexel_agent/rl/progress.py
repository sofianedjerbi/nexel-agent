"""How far the goal still is (#22): the potential of a potential-based shaping reward (Ng, Harada and Russell, 1999).

A tool from bare hands is a long chain (wood, a bench, stone, fibre, then the craft) paid once at its end: few
episodes reach it, and the policy never learns that the chain pays (r43: pursue picked 7% of the times it could,
no tool in 6 tries). The potential is the progress made on the goal since it was set, `UNIT` a step of the recipe
tree, read from the game's recipes (tiles.RECIPES). The shaping reward is the potential's change, and it falls back to
0 when the goal ends, done or not: over a goal it sums to nothing, so it pays no one for trying, nor for taking and
dropping (what is lost is paid back), and the best policy is the one it was without it. It only pays sooner."""

import math

from nexel_worlds.tiles import RECIPES

from .pursue import DROPS

UNIT = 0.05                 # one step of the tree (a unit gathered, a craft, a station put down): a tool is ~12 of them
DEPTH = 4


def left(item: str, inv: dict, at_station, depth: int = 0) -> float:
    """Steps left to make `item` from this pack: each unit it lacks gathered (or made, with its own steps), the station
    made and put down unless one is at hand, then the craft. at_station(name): one within reach."""
    r = RECIPES.get(item)
    if r is None or depth > DEPTH:
        return 1.0
    n = 1.0
    for k, need in r.inputs.items():
        lack = max(0, need - inv.get(k, 0))
        if lack:
            rk = RECIPES.get(k)
            raw = rk is None or k in DROPS                    # (what the land gives is gathered: stone from a rock)
            n += lack if raw else math.ceil(lack / max(1, rk.output.get(k, 1))) * left(k, {}, at_station, depth + 1)
    if r.station and not at_station(r.station):
        n += 1.0 + (0.0 if inv.get(r.station) else left(r.station, inv, at_station, depth + 1))
    return n


def steps_left(goal, inv: dict, at_station, gather_n: int) -> float | None:
    """The goal's steps left, or None for a goal it does not measure (a person's: how near is not progress)."""
    if goal.kind == "craft" and goal.item in RECIPES:
        return left(goal.item, inv, at_station)
    if goal.kind == "gather" and goal.item:
        return float(max(0, goal.base + gather_n - inv.get(goal.item, 0)))
    return None


def potential(goal, inv: dict, at_station, gather_n: int) -> float:
    """UNIT x the steps made since the goal was set (below 0 when it lost ground); 0 for a goal it does not measure."""
    n = steps_left(goal, inv, at_station, gather_n)
    if n is None:
        return 0.0
    if goal.left0 is None:
        goal.left0 = n
    return UNIT * (goal.left0 - n)
