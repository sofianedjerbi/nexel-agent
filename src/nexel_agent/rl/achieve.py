"""Crafter-style achievements: +1 the first time each happens in an episode, over the whole game (a fixed list, so
a score can be told). Each one names something that happened, read from the game: an item came into the pack, an
action went through, the world around says you are in a shelter or a cave, a night was lived through. None says how.
The ones that need a look at the world each step are in rl/milestones.py."""

import math

from ..sim.entities import CREATURES
from ..tiles import ARMOR, BLOCKS, FOOD, RECIPES
from .actions import PLACE

_MADE = {k for r in RECIPES.values() for k in r.output} - {"stone", "hay"}      # (stone, hay also come from the land)
COLLECT = tuple(sorted(({"dirt", "stone_block"} | set(FOOD) | {k for b in BLOCKS for k in b.drops}
                        | {d[0] for c in CREATURES.values() for d in c["drops"]}) - _MADE))
CRAFTS = tuple(RECIPES)
EATS = tuple(FOOD)
DEFEATS = tuple(k for k in CREATURES if k != "wisp")
HOSTILE = ("tusker", "pincer", "strawman", "wyrm", "mossback", "sporeling")
MILESTONES = ("till", "plant", "water_crop", "feed_animal", "milk_cow", "sleep_bed", "wear_any", "wear_armor",
              "place_5_kinds", "place_10_kinds", "shelter", "light_shelter", "house", "toolkit",
              "survive_night", "survive_2_nights", "survive_3_nights", "hale_day", "swim_across",
              "chunks_4", "chunks_9", "chunks_16", "enter_cave", "depth_5", "depth_10",
              "win_hurt", "evade_foe")
ACHIEVEMENTS = tuple([f"collect_{x}" for x in COLLECT] + [f"craft_{x}" for x in CRAFTS] + [f"place_{x}" for x in PLACE]
                     + [f"eat_{x}" for x in EATS] + [f"defeat_{x}" for x in DEFEATS] + list(MILESTONES))


class Ledger:
    """What happened this episode, and the reward for it."""

    def __init__(self):
        self.done: set[str] = set()
        self.counts = {"crafts": 0, "places": 0, "fails": 0, "acts": 0, "idle": 0, "hits": 0, "won": 0, "lost": 0, "fled": 0}
        self.crafted: set[str] = set()
        self.placed: set[str] = set()
        self.foes: dict[int, str] = {}          # who hit me (id -> kind), till the fight is told
        self.killed: set[int] = set()

    def unlock(self, name: str) -> float:
        if name in ACHIEVEMENTS and name not in self.done:
            self.done.add(name)
            return 1.0
        return 0.0

    def result(self, tool: str, args: dict, res: dict, kind: str = "", hp: float = 20) -> float:
        """An action came back."""
        if res.get("status") == "interrupted":
            return 0.0
        self.counts["acts"] += 1
        if not res.get("ok", False):
            self.counts["fails"] += 1
            return 0.0
        r = 0.0
        if tool == "craft":
            self.counts["crafts"] += 1
            self.crafted.add(args.get("recipe"))
            r += self.unlock(f"craft_{args.get('recipe')}")
        elif tool == "place":
            self.counts["places"] += 1
            item = args.get("item")
            self.placed.add(item)
            r += self.unlock(f"place_{item}") + (self.unlock("plant") if item == "seed" else 0)
            r += self.unlock("place_5_kinds") if len(self.placed) >= 5 else 0
            r += self.unlock("place_10_kinds") if len(self.placed) >= 10 else 0
        elif tool == "eat":
            r += self.unlock(f"eat_{args.get('item')}")
        elif tool == "attack" and res.get("killed"):
            self.killed.add(int(args.get("target", -1)))
            r += self.unlock(f"defeat_{kind}") + (self.unlock("win_hurt") if kind in HOSTILE and hp <= 10 else 0)
        elif tool == "dig" and args.get("with") == "hoe":
            r += self.unlock("till")
        elif tool == "water" and any(t.get("ground") == "wet_tilled" for t in (res.get("changed") or {}).get("tiles", [])):
            r += self.unlock("water_crop")
        elif tool in ("feed", "milk", "sleep"):
            r += self.unlock({"feed": "feed_animal", "milk": "milk_cow", "sleep": "sleep_bed"}[tool])
        elif tool == "wear":
            r += self.unlock("wear_any") + (self.unlock("wear_armor") if args.get("item") in ARMOR else 0)
        return r

    def gained(self, before: dict, after: dict, made: dict) -> float:
        r = 0.0
        for k, n in after.items():
            if n > before.get(k, 0) + made.get(k, 0):
                r += self.unlock(f"collect_{k}")
        return r

    def close_fights(self, died: bool):
        for fid in list(self.foes):
            self.counts["won" if fid in self.killed else "lost" if died else "fled"] += 1
        self.foes.clear()


def crafter_score(rates: dict) -> float:
    """Crafter's score: the geometric mean of the success rates (in %), each plus one."""
    xs = [rates.get(a, 0.0) * 100 for a in ACHIEVEMENTS]
    return math.exp(sum(math.log(1 + x) for x in xs) / len(xs)) - 1
