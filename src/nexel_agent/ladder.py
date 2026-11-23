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
    ("tools: axe, pickaxe, hoe", lambda s: all(t in s.ever for t in TOOLS)),
    ("food + campfire", lambda s: "campfire" in s.placed or "cooked_meat" in s.ever),
    ("shelter: walls, door, roof", lambda s: s.placed.get("wood_wall", 0) + s.placed.get("stone_wall", 0) >= 4
     and "door" in s.placed),
    ("farm: till, plant seed", lambda s: "seed" in s.placed),
    ("chest + bed placed", lambda s: "chest" in s.placed and "bed" in s.placed),
    ("sword", lambda s: "sword" in s.ever),
    ("trade/talk with people", lambda s: s.social > 0),
)
TARGETS = {"wood": ["wood"], "workbench placed": ["workbench"], "stone + fiber": ["stone", "fiber"],
           "tools: axe, pickaxe, hoe": list(TOOLS), "food + campfire": ["campfire"],
           "shelter: walls, door, roof": ["wood_wall", "door", "roof"], "farm: till, plant seed": ["hoe", "seed"],
           "chest + bed placed": ["chest", "bed"], "sword": ["sword"], "trade/talk with people": []}


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


def free_dirs(obs) -> list[str]:
    """Sides of you where a thing can be put down: free flat ground, not on a hill's edge."""
    g, h, r = obs["grid"], obs.get("heights"), len(obs["grid"]) // 2
    out = []
    for d, (dx, dy) in DIRS.items():
        x, y = r + dx, r + dy
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


def _walk(obs, tx, ty, act="dig"):
    """'move_to a b, dig D' (or 'dig D' when next to it) for the thing at (tx, ty), or ''."""
    me, reach = obs["self"], reachable(obs)
    best = None
    for d, (dx, dy) in DIRS.items():
        q = (tx - dx - me["x"], ty - dy - me["y"])
        if q in reach and (best is None or reach[q] < best[0]):
            best = (reach[q], tx - dx, ty - dy, d)
    if not best:
        return ""
    return f"{act} {best[3]}" if best[0] == 0 else f"move_to {best[1]} {best[2]}, {act} {best[3]}"


def station_near(obs, station) -> bool:
    g, r = obs["grid"], len(obs["grid"]) // 2
    return any(g[r + dy][r + dx] == STATION_CH.get(station) for dy in range(-2, 3) for dx in range(-2, 3)
               if 0 <= r + dy < len(g) and 0 <= r + dx < len(g))


# ---------------------------------------------------------------- the steps
def place_step(item, obs) -> str:
    d = free_dirs(obs)
    return f"place {d[0]} {item}" if d else f"walk to open flat ground, then place D {item}"


def station_steps(station, have, obs, depth=0) -> list[str]:
    if station_near(obs, station):
        return []
    far = _nearest(obs, STATION_CH.get(station, "?"))
    if far:
        way = _walk(obs, far[1], far[2], "craft")
        return [f"go within 2 tiles of the {station} at ({far[1]},{far[2]})" + (f": {way.split(', craft')[0]}" if way else "")]
    if have.get(station, 0) > 0:
        have[station] -= 1
        return [place_step(station, obs)]
    return steps_for(station, have, obs, 1, depth + 1) + [place_step(station, obs)]


def gather_step(item, short, obs, have, depth=0) -> list[str]:
    srcs = SOURCES.get(item)
    if not srcs:
        if item in FOOD or item in ("meat", "milk", "egg"):
            return [f"get {short} {item} (animals, plants)"]
        return [f"find {short} {item}"]
    chars = "".join(c for c, _ in srcs if not NEEDS_TOOL.get(c) or have.get(NEEDS_TOOL[c]))
    near = _nearest(obs, chars) or _nearest(obs, "".join(c for c, _ in srcs))
    if not near:
        return [f"get {short} {item}: no {' or '.join(n for _, n in srcs)} in view: look_far or walk to the map's edge"]
    tool = NEEDS_TOOL.get(near[3])
    name = dict(srcs)[near[3]]
    if tool and not have.get(tool) and depth < 3:
        return steps_for(tool, have, obs, 1, depth + 1) + [f"dig the {name} with the {tool}"]
    way = _walk(obs, near[1], near[2])
    if way and tool:
        way += f" {tool}"
    return [f"get {short} {item}: {name} at ({near[1]},{near[2]})" + (f": {way}" if way else "")]


def steps_for(name, have, obs, n=1, depth=0) -> list[str]:
    """What making n x `name` still takes, in order (spends from `have`, a copy), the last step 'craft name'."""
    rec = RECIPES.get(name)
    if rec is None:
        return gather_step(name, n, obs, have, depth)
    steps = station_steps(rec.station, have, obs, depth) if rec.station and depth < 3 else []
    for k, per in rec.inputs.items():
        short = per * n - have.get(k, 0)
        if short > 0:
            sub = RECIPES.get(k)
            made = sub.output.get(k, 1) if sub else 1
            if sub and (k not in SOURCES or all(have.get(i, 0) >= c for i, c in sub.inputs.items())) and depth < 3:
                times = -(-short // made)
                steps += steps_for(k, have, obs, times, depth + 1)
                have[k] = have.get(k, 0) + times * made
            else:
                steps += gather_step(k, short, obs, have, depth)
                have[k] = have.get(k, 0) + short
        have[k] = have.get(k, 0) - per * n
    return steps + [f"craft {name}" + (f" {n}" if n > 1 else "")]


def goal_recipe(goal: str | None) -> str | None:
    words = (goal or "").lower().replace("-", " ").replace("_", " ").split()
    words += [a + "_" + b for a, b in zip(words, words[1:])]
    return next((f for w in words for f in (w, w[:-1], w[:-2]) if f in RECIPES or f in SOURCES), None)


def target_for(goal, inv, rec: Record | None):
    """(what to make, why): the goal's recipe unless it is a keeper already carried, else the next milestone's."""
    name = goal_recipe(goal)
    if name and not (name in KEEP and name not in STATION_CH and inv.get(name)) and not (name in SOURCES and inv.get(name, 0) >= PLENTY):
        return name, f"Your goal '{goal}'"
    m = rec.next_milestone() if rec else None
    for t in TARGETS.get(m, []):
        if t in ("workbench", "campfire") and rec and t not in rec.placed:
            return t, f"Next milestone '{m}'"
        if not inv.get(t) and not (t in KEEP and rec and rec.ever.get(t)):
            return t, f"Next milestone '{m}'"
    return None, ""


def progress_lines(obs, goal, rec: Record | None) -> list[str]:
    """The computed 'progress' block of the observation (a few short lines)."""
    inv = obs["self"]["inventory"]
    out = []
    if rec:
        done = [i + 1 for i, (m, _) in enumerate(MILESTONES) if m in rec.reached]
        nxt = rec.next_milestone()
        out.append("Milestones done: " + (", ".join(map(str, done)) or "none") + (f"; next: {nxt}." if nxt else "."))
    name, why = target_for(goal, inv, rec)
    if name:
        if name in STATION_CH:
            steps = station_steps(name, dict(inv), obs) or [f"your {name} is in reach: use it"]
        elif name in RECIPES:
            steps = steps_for(name, dict(inv), obs)
        else:
            steps = gather_step(name, max(1, 10 - inv.get(name, 0)), obs, dict(inv))
        out.append(f"{why}: next " + " > ".join(f"{i + 1}) {s}" for i, s in enumerate(steps[:5])) + ".")
    plenty = [f"{k} {v}" for k, v in inv.items() if v >= PLENTY and k in SOURCES]
    if plenty:
        out.append(f"Plenty of {', '.join(plenty)}: stop gathering that, use it.")
    return out


def placeable_carried(inv) -> list[str]:
    return [k for k in inv if k in PLACEABLE or k in FILLERS]
