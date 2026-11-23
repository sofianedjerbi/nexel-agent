"""Where the player stands on the way a person gets on in this world, worked out from the recipe book and what it
sees, so the model is told facts, not given a script: the milestones it reached, the next missing step for its goal
(the exact command where one is known), what it can make now and what it has plenty of.

The milestones are the usual order of things (wood, a placed workbench, stone and fiber, tools, food, a shelter, a
farm, a chest and a bed, a sword, people); the model picks its goals, this only says what a goal still lacks.
"""

import json
from pathlib import Path

from .policy import DIRS, reachable
from .tiles import BLOCKS, FILLERS, FOOD, PLACEABLE, RECIPES

STATION_CH = {"workbench": "&", "campfire": "^"}
FREE = set(".,_'n-:")                     # dry, flat-ish ground a thing can be put on
PLENTY = 30
TOOLS = ("axe", "pickaxe", "hoe")
KEEP = {"workbench", "campfire", "axe", "pickaxe", "hoe", "spear", "sword", "chest", "bed", "boat", "watering_can"}
MILESTONES = (                            # (short name, done?) over the record: ever = items held, placed = things put down
    ("wood", lambda s: s.ever.get("wood", 0) >= 8),
    ("workbench placed", lambda s: "workbench" in s.placed),
    ("stone + fiber", lambda s: "stone" in s.ever and "fiber" in s.ever),
    ("axe, pickaxe, hoe", lambda s: all(t in s.ever for t in TOOLS)),
    ("campfire", lambda s: "campfire" in s.placed or "cooked_meat" in s.ever),
    ("shelter: walls+door", lambda s: s.placed.get("wood_wall", 0) + s.placed.get("stone_wall", 0) >= 4
     and "door" in s.placed),
    ("farm: seed in tilled soil", lambda s: "seed" in s.placed),
    ("chest + bed", lambda s: "chest" in s.placed and "bed" in s.placed),
    ("sword", lambda s: "sword" in s.ever),
    ("talk/trade", lambda s: s.social > 0),
)
TARGETS = {"wood": ["wood"], "workbench placed": ["workbench"], "stone + fiber": ["stone", "fiber"],
           "axe, pickaxe, hoe": list(TOOLS), "campfire": ["campfire"],
           "shelter: walls+door": ["wood_wall", "door", "roof"], "farm: seed in tilled soil": ["hoe", "seed"],
           "chest + bed": ["chest", "bed"], "sword": ["sword"], "talk/trade": []}


def _sources() -> dict:
    """material -> [(map char, thing)] from what natural blocks drop when dug (tree -> wood, rock -> stone...)."""
    built = set(PLACEABLE.values())
    out: dict = {}
    for b in BLOCKS:
        if b.name in built or not b.drops:
            continue
        for item in b.drops:
            out.setdefault(item, []).append((b.char, b.name.replace("_", " ")))
    return out


SOURCES = _sources()
NEEDS_TOOL = {b.char: b.needs_tool for b in BLOCKS if getattr(b, "needs_tool", None)}


def ladder_text() -> str:
    """For the stable prefix: the milestones, one line."""
    return "Milestones: " + " > ".join(f"{i + 1} {m[0]}" for i, m in enumerate(MILESTONES)) + "."


class Record:
    """What the player has ever held and placed (kept in <agent dir>/progress.json), and when each milestone fell."""

    def __init__(self, path: Path | None = None):
        self.path = path
        d = json.loads(path.read_text()) if path and path.exists() else {}
        self.ever, self.placed = d.get("ever", {}), d.get("placed", {})
        self.social, self.reached = d.get("social", 0), d.get("reached", {})

    def seen(self, inv: dict, t: float = 0.0):
        for k, n in inv.items():
            if n > self.ever.get(k, 0):
                self.ever[k] = n
        self._check(t)

    def did(self, tool: str, args: dict, ok: bool, t: float = 0.0):
        if not ok:
            return
        if tool == "place":
            item = str(args.get("item", ""))
            self.placed[item] = self.placed.get(item, 0) + 1
        elif tool in ("give", "say"):
            self.social += 1
        self._check(t)

    def _check(self, t):
        new = [m for m, done in MILESTONES if m not in self.reached and done(self)]
        for m in new:
            self.reached[m] = round(t, 1)
        if self.path:
            self.path.write_text(json.dumps({"ever": self.ever, "placed": self.placed, "social": self.social,
                                             "reached": self.reached}))

    def next_milestone(self) -> str | None:
        return next((m for m, done in MILESTONES if not done(self)), None)


# ---------------------------------------------------------------- the map
def _level(ch: str) -> int:
    return int(ch) if ch.isdigit() else ord(ch) - 55 if ch.isupper() else 96 - ord(ch) if ch.islower() else 0


def free_dirs(obs, at=(0, 0)) -> list[str]:
    """Sides of you (or of the tile `at` from you) where a thing can be put down: free flat ground, not a hill's edge."""
    g, h, r = obs["grid"], obs.get("heights"), len(obs["grid"]) // 2
    out = []
    for d, (dx, dy) in DIRS.items():
        x, y = r + at[0] + dx, r + at[1] + dy
        if not (1 <= x < len(g) - 1 and 1 <= y < len(g) - 1):
            continue
        if g[y][x] not in FREE:
            continue
        if h and any(_level(h[y + j][x + i]) < _level(h[y][x]) for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if 0 <= y + j < len(h) and 0 <= x + i < len(h)):
            continue
        out.append(d)
    return out


def _nearest(obs, chars):
    """(distance, x, y, char) of the nearest of chars in view."""
    g, me = obs["grid"], obs["self"]
    r = len(g) // 2
    best = None
    for j, row in enumerate(g):
        for i, ch in enumerate(row):
            if ch in chars and (best is None or abs(i - r) + abs(j - r) < best[0]):
                best = (abs(i - r) + abs(j - r), me["x"] - r + i, me["y"] - r + j, ch)
    return best


def station_near(obs, station) -> bool:
    g, r = obs["grid"], len(obs["grid"]) // 2
    return any(g[r + dy][r + dx] == STATION_CH.get(station) for dy in range(-2, 3) for dx in range(-2, 3)
               if 0 <= r + dy < len(g) and 0 <= r + dx < len(g))


# ---------------------------------------------------------------- facts (never steps: the model plans)
def where_to_stand(obs, tx, ty) -> str:
    """'stand at (a,b), it is E of there' (or 'next to you, E') for the thing at (tx, ty), or 'no side reachable'."""
    me, reach = obs["self"], reachable(obs)
    best = None
    for d, (dx, dy) in DIRS.items():
        q = (tx - dx - me["x"], ty - dy - me["y"])
        if q in reach and (best is None or reach[q] < best[0]):
            best = (reach[q], tx - dx, ty - dy, d)
    if not best:
        return "no side of it reachable in view"
    return f"next to you, {best[3]}" if best[0] == 0 else f"stand at ({best[1]},{best[2]}), it is {best[3]} of there"


def station_fact(station, inv, obs) -> str:
    if station_near(obs, station):
        return f"a placed {station} within 2 tiles: yes"
    far = _nearest(obs, STATION_CH.get(station, "?"))
    if far:
        return f"a placed {station} within 2 tiles: no; one is at ({far[1]},{far[2]}), {far[0]} tiles away"
    rec = RECIPES.get(station)
    carried = f"you carry {inv[station]}" if inv.get(station) else "you carry none"
    made = (f"; {station} = " + " + ".join(f"{n} {k}" for k, n in rec.inputs.items())) if rec else ""
    return f"a placed {station} within 2 tiles: none in view; {carried}{made}"


def source_fact(item, obs, inv) -> str:
    """Where `item` comes from, and the nearest such thing in view."""
    srcs = SOURCES.get(item)
    if not srcs:
        return f"{item}: " + ("from animals or plants" if item in FOOD or item in ("meat", "milk", "egg") else
                              ("= " + " + ".join(f"{n} {k}" for k, n in RECIPES[item].inputs.items()) if item in RECIPES
                               else "no source known"))
    names = " or ".join(n for _, n in srcs)
    near = _nearest(obs, "".join(c for c, _ in srcs))
    if not near:
        last = [f"{n} at ({p[0]},{p[1]})" for _, n in srcs if (p := (obs.get("_seen") or {}).get(n))]
        return f"{item} comes from {names}: none in view" + (f"; last seen: {', '.join(last[:2])}" if last else "")
    name = dict(srcs)[near[3]]
    tool = NEEDS_TOOL.get(near[3])
    need = f" (needs a {tool} in hand: you carry {'one' if inv.get(tool) else 'none'})" if tool else ""
    return f"{item} comes from {names}; nearest: {name} at ({near[1]},{near[2]}){need}: {where_to_stand(obs, near[1], near[2])}"


def recipe_facts(name, inv, obs) -> list[str]:
    """What `name` takes, what you have of it, and for what is missing where it comes from."""
    rec = RECIPES[name]
    have = ", ".join(f"{n} {k} (have {inv.get(k, 0)})" for k, n in rec.inputs.items())
    out = [f"{name} = {have}" + (f" @{rec.station}" if rec.station else "")]
    if rec.station:
        out.append(station_fact(rec.station, inv, obs))
    out += [source_fact(k, obs, inv) for k, n in rec.inputs.items() if inv.get(k, 0) < n]
    return out


LADDER = ("workbench", "axe", "pickaxe", "hoe", "campfire", "wood_wall", "door", "roof", "chest", "bed", "sword")


def not_made_line(obs, rec) -> str:
    """The curriculum as facts: what was made already, and what not yet (in the usual order), and which of those
    the pack pays for right now."""
    inv = obs["self"]["inventory"]
    made = [k for k in LADDER if rec.ever.get(k) or k in rec.placed]
    todo = []
    for k in LADDER:
        if k in made and not (k == "workbench" and "workbench" not in rec.placed):
            continue
        if k == "workbench" and rec.ever.get(k):
            todo.append("a placed workbench")
            continue
        r = RECIPES[k]
        pays = all(inv.get(i, 0) >= n for i, n in r.inputs.items())
        here = not r.station or station_near(obs, r.station)
        todo.append(k + (" (you can make it now)" if pays and here else " (your pack pays for it; needs a placed "
                                                                         f"{r.station} near)" if pays else ""))
    return (f"Made so far: {', '.join(made) or 'nothing from the recipe book'}. Not made yet: "
            + ", ".join(todo[:6]) + ".")


def goal_recipe(goal: str | None) -> str | None:
    words = (goal or "").lower().replace("-", " ").replace("_", " ").split()
    words += [a + "_" + b for a, b in zip(words, words[1:])]
    return next((f for w in words for f in (w, w[:-1], w[:-2]) if f in RECIPES or f in SOURCES), None)


def progress_lines(obs, goal, rec: Record | None) -> list[str]:
    """Facts for the goal: the milestones reached, and what the goal's recipe or material takes and where it is."""
    inv = obs["self"]["inventory"]
    out = []
    if rec:
        out.append(not_made_line(obs, rec))
    name = goal_recipe(goal)
    if name in RECIPES:
        out.append("For your goal: " + "; ".join(recipe_facts(name, inv, obs)) + ".")
    elif name:
        out.append("For your goal: " + source_fact(name, obs, inv) + ".")
    return out


def placeable_carried(inv) -> list[str]:
    return [k for k in inv if k in PLACEABLE or k in FILLERS]
