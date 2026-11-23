"""What a learning player sees: the agent API's observation dict made into arrays (the actions: rl/actions.py).

Both the training env (sim in-process) and the live body (the server's observe) go through here, so the policy
sees the same thing in both. Only numpy (it runs in the game's Python, which has no torch or gymnasium)."""

import math

import numpy as np

from megastructure.sim.constants import DAY_TICKS, VIEW_R
from megastructure.sim.entities import CREATURES
from megastructure.sim.specs import LEGEND
from megastructure.tiles import BLOCKS, FOOD, RECIPES
from .actions import ACTIONS, DIRS, LEVEL, N_ACTIONS, action_name, legal, person, to_command  # noqa: F401 (one place to import from)
from .goals import KINDS as GOAL_KINDS

SIDE = 2 * VIEW_R + 1
CHARS = "".join(sorted(set(LEGEND) | set("@P$~")))
CHAR_ID = {ch: i + 1 for i, ch in enumerate(CHARS)}           # 0: a character the legend does not know
N_CHARS = len(CHARS) + 1


def _items():
    items = {"dirt", "stone_block"} | set(FOOD)
    for r in RECIPES.values():
        items |= set(r.inputs) | set(r.output)
    for b in BLOCKS:
        items |= set(b.drops)
    for c in CREATURES.values():
        items |= {d[0] for d in c["drops"]}
    return tuple(sorted(items))


ITEMS = _items()                     # every thing a pack can hold: what is made, dug, picked, dropped
ITEM_ID = {k: i for i, k in enumerate(ITEMS)}
KINDS = tuple(CREATURES) + ("agent",)
N_NEAR = 3
BODY = 16
SOCIAL_FLAGS = 3                     # spoken to just now, asked for something I carry, given something (rl/social.py)
ACTIONS_V1 = 179                     # the actions an r12 policy knew: their last-action slots stay where they were
GOAL = len(GOAL_KINDS) + len(ITEMS) + 4      # the goal: kind, item, the person (dx, dy, hp, in view) (rl/goals.py)
# (each part that came later is appended, so an older policy grows into a newer one: rl/train.py init_from)
PEOPLE = 2 * N_NEAR                  # per nearest being: hit me lately, I hit it lately (rl/social.py)
ACTIONS_V2 = 181                     # the actions an r13-r15 policy knew (their slots before the goal)
VEC = (BODY + len(ITEMS) + N_NEAR * (3 + len(KINDS)) + ACTIONS_V1 + SOCIAL_FLAGS + (ACTIONS_V2 - ACTIONS_V1) + GOAL + PEOPLE
       + (N_ACTIONS - ACTIONS_V2))

_GRID_LUT = np.zeros(128, np.uint8)
for _ch, _i in CHAR_ID.items():
    _GRID_LUT[ord(_ch)] = _i
_LEVEL_LUT = np.zeros(128, np.float32)
for _ch, _v in LEVEL.items():
    _LEVEL_LUT[ord(_ch)] = _v


def encode(obs: dict, facing: str = "S", last_action: int = 0, last_ok: float = 0.0, hurt: float = 0.0, social=(0, 0, 0),
           goal=None, people=None) -> dict:
    """grid: (21, 21) char ids; height: (21, 21) the land's height minus yours, in layers / 4, clipped to [-1, 1];
    vec: body, clock, last action and how it went, pack counts (log), the three nearest beings."""
    rows = obs["grid"]
    raw = np.frombuffer("".join(rows).encode("ascii", "replace"), np.uint8).reshape(SIDE, SIDE) & 127
    grid = _GRID_LUT[raw]
    me = obs["self"]
    if obs.get("heights"):
        hr = np.frombuffer("".join(obs["heights"]).encode("ascii", "replace"), np.uint8).reshape(SIDE, SIDE) & 127
        lv = _LEVEL_LUT[hr]
        height = np.clip((lv - lv[VIEW_R, VIEW_R]) / 4.0, -1, 1).astype(np.float32)
    else:
        height = np.zeros((SIDE, SIDE), np.float32)
    v = np.zeros(VEC, np.float32)
    tod = (obs["tick"] % DAY_TICKS) / DAY_TICKS
    inv = me.get("inventory", {})
    v[:BODY] = (me["hp"] / 20, me["hunger"] / 20, me.get("saturation", 0) / 20, math.sin(2 * math.pi * tod),
                math.cos(2 * math.pi * tod), float(tod >= 0.75), float(bool(me.get("swimming"))), float(bool(me.get("in_cave"))),
                float(bool(me.get("sheltered"))), float(me.get("action") is not None), last_ok, min(hurt / 4, 2.0),
                *(float(facing == d) for d in DIRS))
    o = BODY
    for j, it in enumerate(ITEMS):
        v[o + j] = math.log1p(inv.get(it, 0)) / 3
    o += len(ITEMS)
    for e in obs.get("entities", [])[:N_NEAR]:
        v[o:o + 3] = (e["dx"] / VIEW_R, e["dy"] / VIEW_R, e.get("hp", 0) / 12)
        k = e.get("kind")
        if k in KINDS:
            v[o + 3 + KINDS.index(k)] = 1
        o += 3 + len(KINDS)
    o = BODY + len(ITEMS) + N_NEAR * (3 + len(KINDS))
    late = N_ACTIONS - ACTIONS_V2
    v[o + last_action if last_action < ACTIONS_V1 else o + SOCIAL_FLAGS + last_action if last_action < ACTIONS_V2
      else VEC - late + last_action - ACTIONS_V2] = 1
    v[o + ACTIONS_V1:o + ACTIONS_V1 + SOCIAL_FLAGS] = social
    if people is not None:
        v[VEC - late - PEOPLE:VEC - late] = people
    g = VEC - late - PEOPLE - GOAL
    if goal is not None and goal.kind != "free":
        v[g + GOAL_KINDS.index(goal.kind)] = 1
        if goal.item in ITEM_ID:
            v[g + len(GOAL_KINDS) + ITEM_ID[goal.item]] = 1
        p = person(obs, goal.person)
        if p is not None:
            v[g + GOAL - 4:g + GOAL] = (p["dx"] / VIEW_R, p["dy"] / VIEW_R, p.get("hp", 0) / 20, 1.0)
    return {"grid": grid, "height": height, "vec": v}
