"""What a learning player can do: the action table (primitive game tools), what each means as a tool call, and which
ones the game would refuse outright, from what the observation shows (the legal mask: the game's rules, never a
preference among the legal ones). Nothing here decides anything: `with` is a tool you carry or bare hands, attack
swings the strongest weapon you carry at the nearest being, feed gives the first food you carry that animals eat,
wear puts on the first piece you carry (armor first), give hands over what the goal or that person's ask names, ask
names what the goal needs - the "how" of one action. Only numpy (it runs in the game's Python)."""

import numpy as np

from megastructure.sim.constants import FOOD_MAX, LAYER, VIEW_R
from . import ring
from .facts import within
from megastructure.tiles import ARMOR, BLOCKS, FEED, FILLERS, FOOD, GROUNDS, PLACEABLE, RECIPES, REACH, USE_R, WEAPONS, WEAR

DIRS = ("N", "E", "S", "W")
STEP = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
PLACE = tuple(sorted(set(PLACEABLE) | set(FILLERS) | {"roof", "stairs", "ladder"}))
CRAFT = tuple(RECIPES)
EAT = tuple(FOOD)
FARM = ("chicken", "cow", "chick", "calf")
WEAR_ORDER = tuple(sorted(WEAR, key=lambda w: (-ARMOR.get(w, 0), w)))
GROUND_CH = {g.char for g in GROUNDS} | {"~", "$", "@"}          # (a tile with nothing standing on it)
SOLID_CH = {b.char for b in BLOCKS if b.solid and b.name != "none"}
BLOCK_NAME = {b.char: b.name for b in BLOCKS}


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
    acts += [("say", None, None), ("give", None, None)]     # (appended: a policy trained before them keeps its indices)
    acts += [("ask", None, None), ("approach", None, None)]
    acts += [("attack", None, k) for k in (1, 2)]     # attack the 2nd, 3rd nearest being (the obs's slots); plain attack: the 1st
    acts += [("attack", None, "back")]                # attack whoever hit me last (rl/social.py)
    acts += [("ring", None, None)]                    # walls round me from the pack (rl/ring.py)
    return tuple(acts)


ACTIONS = _actions()
N_ACTIONS = len(ACTIONS)
RING_I = next(i for i, a in enumerate(ACTIONS) if a[0] == "ring")


def action_name(i: int) -> str:
    kind, d, arg = ACTIONS[i]
    return "_".join(str(p) for p in (kind, d, arg) if p)


def _near(obs, kinds, r=USE_R):
    return [e for e in obs.get("entities", []) if e.get("kind") in kinds and max(abs(e["dx"]), abs(e["dy"])) <= r]


def _people(obs, r):
    return [e for e in obs.get("entities", []) if e.get("kind") == "agent" and max(abs(e["dx"]), abs(e["dy"])) <= r]


def person(obs, name):
    return next((e for e in obs.get("entities", []) if e.get("kind") == "agent" and e.get("name") == name), None) if name else None


def wanted(obs, goal) -> str | None:
    """The item an ask names: the goal's (gather, ask), or the first input a craft goal still lacks."""
    if goal is None or not goal.item:
        return None
    if goal.kind in ("gather", "ask"):
        return goal.item
    if goal.kind == "craft" and goal.item in RECIPES:
        inv = obs["self"].get("inventory", {})
        return next((k for k, n in RECIPES[goal.item].inputs.items() if inv.get(k, 0) < n), None)
    return None


def handed(obs, goal, social, to) -> str | None:
    """What give hands to that person: the goal's item (give goal, that person), else what they asked me for."""
    inv = obs["self"].get("inventory", {})
    if goal is not None and goal.kind == "give" and goal.person == to and goal.item and inv.get(goal.item):
        return goal.item
    ask = social.ask_open(obs["tick"]) if social else None
    return ask if ask and social.asked_by == to and inv.get(ask) else None


def to_command(i: int, obs: dict, facing: str, social=None, goal=None):
    """The game tool and args for action i (None: keep doing what you do). Words are placeholders: on the live
    server the LLM writes them."""
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
    if kind == "attack":                                  # (the being in that slot of the obs: nearest first)
        ents = obs.get("entities", [])
        if arg == "back":
            e = social.hitter(obs) if social else None
        else:
            e = ents[arg or 0] if len(ents) > (arg or 0) else None
        w = max((w for w in WEAPONS if inv.get(w)), key=lambda w: WEAPONS[w], default=None)
        fight = arg == "back" or bool(e and social and e.get("kind") == "agent" and social.hit_me_lately(e.get("name"), obs["tick"]))
        return "attack", {"target": e["id"] if e else -1, **({"with": w} if w else {}), **({"fight": True} if fight else {})}
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
    if kind == "ring":
        p = ring.plan(obs, GROUND_CH) or []
        return "ring", {"pieces": [(me["x"] + dx, me["y"] + dy, it) for dx, dy, it in p]}
    if kind == "say":
        return "say", {"text": "hi"}
    if kind == "ask":
        return "say", {"text": f"could you spare some {wanted(obs, goal) or 'help'}?", "ask": wanted(obs, goal)}
    if kind == "give":
        p = next((e for e in _people(obs, USE_R) if handed(obs, goal, social, e.get("name"))), None)
        return "give", {"target": p["id"] if p else -1, "item": (handed(obs, goal, social, p.get("name")) if p else "") or "", "n": 1}
    if kind == "approach":
        p = person(obs, goal.person if goal else None)
        return "move_to", {"x": me["x"] + (p["dx"] if p else 0), "y": me["y"] + (p["dy"] if p else 0)}
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


def _front(obs, d):
    """(the char in front, its height over mine, on a hill's rim: a lower tile beside it)"""
    dx, dy = STEP[d]
    ch = obs["grid"][VIEW_R + dy][VIEW_R + dx]
    hs = obs.get("heights")
    if not hs:
        return ch, 0, False
    lv = lambda x, y: LEVEL.get(hs[y][x], 0)          # noqa: E731
    fx, fy = VIEW_R + dx, VIEW_R + dy
    rim = any(lv(fx + i, fy + j) < lv(fx, fy) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j)
    return ch, lv(fx, fy) - lv(VIEW_R, VIEW_R), rim


def legal(obs: dict, facing: str = "S", social=None, goal=None) -> np.ndarray:
    """Which actions the game would not refuse outright, from what the observation shows: what you carry, a station
    or a bed within reach, a being in reach (height counted), what is in front (a block to dig, bare ground only a
    hoe tills and only a pickaxe cuts into, a hill's rim where nothing is built, a tile already taken), a full belly."""
    me = obs["self"]
    inv = me.get("inventory", {})
    near = {ch for row in obs["grid"][VIEW_R - USE_R:VIEW_R + USE_R + 1] for ch in row[VIEW_R - USE_R:VIEW_R + USE_R + 1]}
    m = np.ones(N_ACTIONS, bool)
    worn = set((me.get("wearing") or {}).values())
    fronts = {d: _front(obs, d) for d in DIRS}
    fch, frel, frim = fronts.get(facing, fronts["S"])
    weapon = max((w for w in WEAPONS if inv.get(w)), key=lambda w: WEAPONS[w], default=None)
    hs = obs.get("heights")
    for i, (kind, d, arg) in enumerate(ACTIONS):
        ch, rel, rim = fronts[d] if d else (fch, frel, frim)
        if kind == "step":
            m[i] = rel <= 0 and ch not in SOLID_CH
        elif kind == "dig_at":            # (on the level, the game reads it as tilling and refuses)
            m[i] = rel >= arg
        elif kind == "climb":             # (a wall of land in front, one layer or more over your feet)
            m[i] = rel >= 1 and ch not in SOLID_CH
        elif kind == "dig":
            bare = ch in GROUND_CH
            if arg and not inv.get(arg):
                m[i] = False
            elif bare:                    # (no block there: a hoe tills it on the level, a pickaxe cuts into land)
                m[i] = (arg == "hoe" and rel == 0) or (arg == "pickaxe" and rel != 0)
            else:
                m[i] = arg != "hoe" and not (BLOCK_NAME.get(ch) == "ore" and arg != "pickaxe")
        elif kind == "attack":
            ents = obs.get("entities", [])
            e = (social.hitter(obs) if social else None) if arg == "back" else (ents[arg or 0] if len(ents) > (arg or 0) else None)
            if e is not None and e.get("kind") != "wisp":
                dz = (LEVEL.get(hs[VIEW_R + e["dy"]][VIEW_R + e["dx"]], 0) - LEVEL.get(hs[VIEW_R][VIEW_R], 0)) if hs and abs(e["dx"]) <= VIEW_R and abs(e["dy"]) <= VIEW_R else 0
                m[i] = within(e, REACH.get(weapon, REACH[None]) + (e.get("kind") == "mossback"), dz * LAYER)   # (the server's reach)
            else:
                m[i] = False
        elif kind == "pick_up":
            m[i] = any(it["dx"] == 0 and it["dy"] == 0 for it in obs.get("items", []))
        elif kind == "ring":
            m[i] = ring.plan(obs, GROUND_CH) is not None
        elif kind == "place":
            m[i] = bool(inv.get(arg)) and not frim and (fch in GROUND_CH or BLOCK_NAME.get(fch) == "flower")
            if arg in ("stairs", "ladder"):
                m[i] = bool(inv.get(arg)) and frel >= 2
        elif kind == "eat":
            m[i] = bool(inv.get(arg)) and me.get("hunger", 0) < FOOD_MAX
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
            m[i] = bool(_people(obs, VIEW_R)) and (social is None or social.can_talk(obs["tick"]))
        elif kind == "ask":
            m[i] = bool(_people(obs, VIEW_R)) and wanted(obs, goal) is not None and (social is None or social.can_talk(obs["tick"]))
        elif kind == "give":
            m[i] = any(handed(obs, goal, social, e.get("name")) for e in _people(obs, USE_R))
        elif kind == "approach":
            p = person(obs, goal.person if goal else None)
            m[i] = p is not None and max(abs(p["dx"]), abs(p["dy"])) > 1
    return m
