"""What a learning player can do: the action table (primitive game tools), what each means as a tool call, and which
ones the game would refuse outright (the legal mask). Nothing here decides anything: `with` is a tool you carry or
bare hands, attack swings the strongest weapon you carry at the nearest being, feed gives the first food you carry
that animals eat, wear puts on the first piece you carry (armor first, strongest first) - the "how" of one action.
Only numpy (it runs in the game's Python)."""

import numpy as np

from megastructure.sim.constants import VIEW_R
from megastructure.tiles import ARMOR, BLOCKS, FEED, FILLERS, FOOD, PLACEABLE, RECIPES, USE_R, WEAPONS, WEAR

DIRS = ("N", "E", "S", "W")
STEP = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
PLACE = tuple(sorted(set(PLACEABLE) | set(FILLERS) | {"roof", "stairs", "ladder"}))
CRAFT = tuple(RECIPES)
EAT = tuple(FOOD)
FARM = ("chicken", "cow", "chick", "calf")
WEAR_ORDER = tuple(sorted(WEAR, key=lambda w: (-ARMOR.get(w, 0), w)))


def _actions():
    acts = [("noop", None, None)]
    acts += [("step", d, None) for d in DIRS]
    acts += [("jump", d, None) for d in DIRS]
    acts += [("dig", d, w) for w in (None, "axe", "pickaxe", "hoe") for d in DIRS]
    acts += [("dig_at", d, h) for h in (1, 2) for d in DIRS]          # the land in front, a layer or two over your feet
    acts += [("climb", d, None) for d in DIRS]      # a step up a wall: the land in front at head height dug, then a jump
    acts += [("dig_down", None, None), ("pick_up", None, None), ("attack", None, None)]
    acts += [("water", None, None), ("sleep", None, None), ("feed", None, None), ("milk", None, None), ("wear", None, None)]
    acts += [("place", None, it) for it in PLACE]
    acts += [("craft", None, r) for r in CRAFT]
    acts += [("eat", None, f) for f in EAT]
    acts += [("say", None, None), ("give", None, None)]     # (last: a policy trained before them keeps its indices)
    return tuple(acts)


ACTIONS = _actions()
N_ACTIONS = len(ACTIONS)


def action_name(i: int) -> str:
    kind, d, arg = ACTIONS[i]
    return "_".join(str(p) for p in (kind, d, arg) if p)


def _near(obs, kinds, r=USE_R):
    return [e for e in obs.get("entities", []) if e.get("kind") in kinds and max(abs(e["dx"]), abs(e["dy"])) <= r]


def _people(obs, r):
    return [e for e in obs.get("entities", []) if e.get("kind") == "agent" and max(abs(e["dx"]), abs(e["dy"])) <= r]


def to_command(i: int, obs: dict, facing: str, social=None):
    """The game tool and args for action i (None: keep doing what you do). say's words are a placeholder: on the
    live server the LLM's chat path writes them; give hands the nearest person what social.gift says."""
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
    if kind == "climb":
        return "climb", {"dir": d, "z": int(me.get("z", 0)) + 2}
    if kind in ("dig_down", "pick_up"):
        return kind, {}
    if kind == "attack":
        near = [e for e in obs.get("entities", []) if e.get("kind") != "wisp"]
        w = max((w for w in WEAPONS if inv.get(w)), key=lambda w: WEAPONS[w], default=None)
        return "attack", {"target": near[0]["id"] if near else -1, **({"with": w} if w else {})}
    if kind == "water":
        return "water", {"dir": facing, "with": "watering_can"}
    if kind == "sleep":
        return "sleep", {"dir": facing}
    if kind in ("feed", "milk"):
        near = _near(obs, FARM if kind == "feed" else ("cow",))
        food = next((f for f in FEED if inv.get(f)), None)
        return kind, {"target": near[0]["id"] if near else -1, **({"item": food} if kind == "feed" and food else {})}
    if kind == "wear":
        worn = set((me.get("wearing") or {}).values())
        return "wear", {"item": next((w for w in WEAR_ORDER if inv.get(w) and w not in worn), "")}
    if kind == "place":
        return "place", {"dir": facing, "item": arg}
    if kind == "say":
        return "say", {"text": "hi"}
    if kind == "give":
        near = _people(obs, USE_R)
        item = social.gift(obs) if social else max(inv, key=inv.get, default=None)
        return "give", {"target": near[0]["id"] if near else -1, "item": item or "", "n": 1}
    if kind == "craft":
        return "craft", {"recipe": arg}
    return "eat", {"item": arg}


def _level(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - 55
    return 96 - ord(ch)


LEVEL = {chr(c): _level(chr(c)) for c in range(48, 123) if chr(c).isalnum()}
_STATION_CH = {b.name: b.char for b in BLOCKS}


def legal(obs: dict) -> np.ndarray:
    """Which actions the game would not refuse outright (what you carry, a station or a bed within reach, a being
    in reach, land in front that high): the rules' own preconditions, never a preference among the legal ones."""
    me = obs["self"]
    inv = me.get("inventory", {})
    near = {ch for row in obs["grid"][VIEW_R - USE_R:VIEW_R + USE_R + 1] for ch in row[VIEW_R - USE_R:VIEW_R + USE_R + 1]}
    m = np.ones(N_ACTIONS, bool)
    hs = obs.get("heights")
    worn = set((me.get("wearing") or {}).values())
    for i, (kind, d, arg) in enumerate(ACTIONS):
        if kind == "dig_at":            # (on the level, the game reads it as tilling and refuses)
            dx, dy = STEP[d]
            m[i] = bool(hs) and LEVEL.get(hs[VIEW_R + dy][VIEW_R + dx], 0) - LEVEL.get(hs[VIEW_R][VIEW_R], 0) >= arg
        elif kind == "climb":           # (a wall in front: one layer or more over your feet)
            dx, dy = STEP[d]
            m[i] = bool(hs) and LEVEL.get(hs[VIEW_R + dy][VIEW_R + dx], 0) - LEVEL.get(hs[VIEW_R][VIEW_R], 0) >= 1
        elif kind == "dig" and arg:
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
        elif kind == "water":
            m[i] = bool(inv.get("watering_can"))
        elif kind == "sleep":
            m[i] = _STATION_CH["bed"] in near
        elif kind == "feed":
            m[i] = bool(_near(obs, FARM)) and any(inv.get(f) for f in FEED)
        elif kind == "milk":
            m[i] = bool(_near(obs, ("cow",)))
        elif kind == "wear":
            m[i] = any(inv.get(w) and w not in worn for w in WEAR)
        elif kind == "say":
            m[i] = bool(_people(obs, VIEW_R))
        elif kind == "give":
            m[i] = bool(_people(obs, USE_R)) and any(inv.values())
    return m
