"""What a learning player sees and can do: the agent API's observation dict made into arrays, and the action table.

Both the training env (sim in-process) and the live body (the server's observe) go through here, so the policy
sees the same thing in both. Nothing here decides anything: it only says what the obs is and what an action means.
Only numpy (it runs in the game's Python, which has no torch or gymnasium)."""

import math

import numpy as np

from ..sim.constants import DAY_TICKS, VIEW_R
from ..sim.specs import LEGEND
from ..tiles import BLOCKS, FOOD, RECIPES, USE_R, WEAPONS

SIDE = 2 * VIEW_R + 1
DIRS = ("N", "E", "S", "W")
CHARS = "".join(sorted(set(LEGEND) | set("@P$~")))
CHAR_ID = {ch: i + 1 for i, ch in enumerate(CHARS)}           # 0: a character the legend does not know
N_CHARS = len(CHARS) + 1

ITEMS = ("wood", "stone", "stone_block", "dirt", "fiber", "plank", "workbench", "campfire", "chest", "bed", "door",
         "fence", "wood_wall", "stone_wall", "path", "ladder", "stairs", "roof", "torch", "rope", "axe", "pickaxe",
         "hoe", "spear", "sword", "ore", "ingot", "berries", "fruit", "seed", "flower", "mushroom", "meat",
         "cooked_meat", "wheat", "bread", "egg", "milk", "hay", "shell_plate", "sign", "basket")
PLACE = ("workbench", "campfire", "chest", "bed", "door", "fence", "wood_wall", "stone_wall", "dirt", "stone_block",
         "plank", "path", "torch")
CRAFT = ("workbench", "plank", "stone", "campfire", "torch", "rope", "axe", "pickaxe", "hoe", "spear", "cooked_meat",
         "bread", "ingot", "sword", "door", "fence", "chest", "wood_wall", "stone_wall", "bed", "ladder", "stairs",
         "path", "roof", "sign", "basket")
EAT = ("berries", "fruit", "meat", "cooked_meat", "mushroom", "bread", "egg", "milk")
KINDS = ("chicken", "cow", "chick", "calf", "tusker", "pincer", "strawman", "wisp", "wyrm", "mossback", "sporeling", "agent")


def _actions():
    acts = [("noop", None, None)]
    acts += [("step", d, None) for d in DIRS]
    acts += [("jump", d, None) for d in DIRS]
    acts += [("dig", d, w) for w in (None, "axe", "pickaxe") for d in DIRS]
    acts += [("dig_at", d, h) for h in (1, 2) for d in DIRS]          # the land in front, a layer or two over your feet
    acts += [("dig_down", None, None), ("pick_up", None, None), ("attack", None, None)]
    acts += [("place", None, it) for it in PLACE]
    acts += [("craft", None, r) for r in CRAFT]
    acts += [("eat", None, f) for f in EAT]
    return tuple(acts)


ACTIONS = _actions()
N_ACTIONS = len(ACTIONS)
N_NEAR = 3
VEC = 16 + len(ITEMS) + N_NEAR * (3 + len(KINDS)) + N_ACTIONS


def action_name(i: int) -> str:
    kind, d, arg = ACTIONS[i]
    return "_".join(str(p) for p in (kind, d, arg) if p)


def to_command(i: int, obs: dict, facing: str):
    """The game tool and args for action i (None: keep doing what you do). `with` is a tool you carry or bare hands;
    attack swings the strongest weapon you carry (WEAPONS) at the nearest being."""
    kind, d, arg = ACTIONS[i]
    me = obs["self"]
    inv = me.get("inventory", {})
    if kind == "noop":
        return None
    if kind in ("step", "jump"):
        return kind, {"dir": d}
    if kind == "dig":
        return "dig", {"dir": d, **({"with": arg} if arg else {})}
    if kind == "dig_at":
        return "dig", {"dir": d, "z": int(me.get("z", 0)) + arg}
    if kind == "dig_down":
        return "dig_down", {}
    if kind == "pick_up":
        return "pick_up", {}
    if kind == "attack":
        near = [e for e in obs.get("entities", []) if e.get("kind") != "wisp"]
        if not near:
            return "attack", {"target": -1}
        w = max((w for w in WEAPONS if inv.get(w)), key=lambda w: WEAPONS[w], default=None)
        return "attack", {"target": near[0]["id"], **({"with": w} if w else {})}
    if kind == "place":
        return "place", {"dir": facing, "item": arg}
    if kind == "craft":
        return "craft", {"recipe": arg}
    return "eat", {"item": arg}


def _level(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - 55
    return 96 - ord(ch)


_LEVEL = {chr(c): _level(chr(c)) for c in range(48, 123) if chr(c).isalnum()}
_GRID_LUT = np.zeros(128, np.uint8)
for _ch, _i in CHAR_ID.items():
    _GRID_LUT[ord(_ch)] = _i
_LEVEL_LUT = np.zeros(128, np.float32)
for _ch, _v in _LEVEL.items():
    _LEVEL_LUT[ord(_ch)] = _v


def encode(obs: dict, facing: str = "S", last_action: int = 0, last_ok: float = 0.0, hurt: float = 0.0) -> dict:
    """grid: (21, 21) char ids; height: (21, 21) the land's height minus yours, in layers / 4, clipped to [-1, 1];
    vec: body, clock, last action and how it went, pack counts (log), the three nearest beings."""
    rows = obs["grid"]
    raw = np.frombuffer("".join(rows).encode("ascii", "replace"), np.uint8).reshape(SIDE, SIDE) & 127
    grid = _GRID_LUT[raw]
    me = obs["self"]
    if obs.get("heights"):
        hr = np.frombuffer("".join(obs["heights"]).encode("ascii", "replace"), np.uint8).reshape(SIDE, SIDE) & 127
        lv = _LEVEL_LUT[hr]
        mine = lv[VIEW_R, VIEW_R]
        height = np.clip((lv - mine) / 4.0, -1, 1).astype(np.float32)
    else:
        height = np.zeros((SIDE, SIDE), np.float32)
    v = np.zeros(VEC, np.float32)
    tod = (obs["tick"] % DAY_TICKS) / DAY_TICKS
    inv = me.get("inventory", {})
    v[:16] = (me["hp"] / 20, me["hunger"] / 20, me.get("saturation", 0) / 20, math.sin(2 * math.pi * tod),
              math.cos(2 * math.pi * tod), float(tod >= 0.75), float(bool(me.get("swimming"))), float(bool(me.get("in_cave"))),
              float(bool(me.get("sheltered"))), float(me.get("action") is not None), last_ok, min(hurt / 4, 2.0),
              *(float(facing == d) for d in DIRS))
    o = 16
    for j, it in enumerate(ITEMS):
        v[o + j] = math.log1p(inv.get(it, 0)) / 3
    o += len(ITEMS)
    for e in obs.get("entities", [])[:N_NEAR]:
        v[o:o + 3] = (e["dx"] / VIEW_R, e["dy"] / VIEW_R, e.get("hp", 0) / 12)
        k = e.get("kind")
        if k in KINDS:
            v[o + 3 + KINDS.index(k)] = 1
        o += 3 + len(KINDS)
    o = VEC - N_ACTIONS
    v[o + last_action] = 1
    return {"grid": grid, "height": height, "vec": v}


FOODS = tuple(FOOD)
_STATION_CH = {b.name: b.char for b in BLOCKS}


def legal(obs: dict) -> np.ndarray:
    """Which actions the game would not refuse outright (what you carry, a station within reach, a being in view):
    the rules' own preconditions, never a preference among the legal ones."""
    me = obs["self"]
    inv = me.get("inventory", {})
    near = {ch for row in obs["grid"][VIEW_R - USE_R:VIEW_R + USE_R + 1] for ch in row[VIEW_R - USE_R:VIEW_R + USE_R + 1]}
    m = np.ones(N_ACTIONS, bool)
    for i, (kind, d, arg) in enumerate(ACTIONS):
        if kind == "dig" and arg:
            m[i] = bool(inv.get(arg))
        elif kind == "attack":
            m[i] = any(e.get("kind") != "wisp" and abs(e["dx"]) <= 4 and abs(e["dy"]) <= 4 for e in obs.get("entities", []))
        elif kind == "pick_up":
            m[i] = any(it["dx"] == 0 and it["dy"] == 0 for it in obs.get("items", []))
        elif kind in ("place", "eat"):
            m[i] = bool(inv.get(arg))
        elif kind == "craft":
            r = RECIPES[arg]
            m[i] = all(inv.get(k, 0) >= n for k, n in r.inputs.items()) and (not r.station or _STATION_CH[r.station] in near)
    return m
