"""Crafter-style achievements: +1 the first time each happens in an episode. A fixed list, so a score can be told.
No achievement says how to get it; they only name things that happened (an item came into the pack, a craft or a
place or a meal went through, a being was killed, a night was lived through)."""

import math

COLLECT = ("wood", "fiber", "stone", "stone_block", "dirt", "berries", "seed", "fruit", "flower", "mushroom", "ore",
           "meat", "wheat", "egg", "milk", "hay")
CRAFTS = ("workbench", "plank", "stone", "campfire", "torch", "rope", "axe", "pickaxe", "hoe", "spear", "cooked_meat",
          "bread", "ingot", "sword", "door", "fence", "chest", "wood_wall", "stone_wall", "bed", "ladder", "stairs", "path")
PLACES = ("workbench", "campfire", "chest", "bed", "door", "fence", "wood_wall", "stone_wall", "dirt", "stone_block",
          "plank", "path", "torch")
EATS = ("berries", "fruit", "meat", "cooked_meat", "mushroom", "bread", "egg", "milk")
DEFEATS = ("tusker", "pincer", "strawman", "chicken", "cow", "sporeling")

ACHIEVEMENTS = tuple([f"collect_{x}" for x in COLLECT] + [f"craft_{x}" for x in CRAFTS] + [f"place_{x}" for x in PLACES]
                     + [f"eat_{x}" for x in EATS] + [f"defeat_{x}" for x in DEFEATS] + ["survive_night"])
HOSTILE = ("tusker", "pincer", "strawman", "wyrm", "mossback", "sporeling")


class Ledger:
    """What happened this episode, and the reward for it."""

    def __init__(self):
        self.done: set[str] = set()
        self.counts = {"crafts": 0, "places": 0, "fails": 0, "acts": 0, "idle": 0, "hits": 0, "won": 0, "lost": 0, "fled": 0}
        self.crafted: set[str] = set()
        self.foes: dict[int, str] = {}          # who hit me (id -> kind), till the fight is told
        self.killed: set[int] = set()

    def unlock(self, name: str) -> float:
        if name in ACHIEVEMENTS and name not in self.done:
            self.done.add(name)
            return 1.0
        return 0.0

    def result(self, tool: str, args: dict, res: dict, kind: str = "") -> float:
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
            r += self.unlock(f"place_{args.get('item')}")
        elif tool == "eat":
            r += self.unlock(f"eat_{args.get('item')}")
        elif tool == "attack" and res.get("killed"):
            tid = int(args.get("target", -1))
            self.killed.add(tid)
            r += self.unlock(f"defeat_{kind}")
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
