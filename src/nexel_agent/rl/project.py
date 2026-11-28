"""A build is a project the body works at to its end (#87). The model plans, code executes it (Voyager, Mindcraft,
CraftUtopia: the model names what, skills run to the end): each decision, what the pack allows is laid (the build
verb, rl/build.py), else the raw material the unlaid pieces lack most is gathered (the gather verb, rl/pursue.py),
read from the game's own recipes down to what the land gives. Live trials 4 to 6 built nothing drawn: the builder
stopped when the pack was short, and gathering the hundreds of pieces a house takes was left to the mind's goals,
set every 20 s, with the learned body (paid for crafting every thing once) acting between them."""

from collections import Counter

import numpy as np

from nexel_worlds.tiles import BLOCKS, PLACEABLE, RECIPES

from .actions import ACTIONS
from .goals import Goal
from .pursue import DROPS

BUILD = next(k for k, a in enumerate(ACTIONS) if a[0] == "build")
UNASKED = ("craft", "place", "dig", "dig_at", "dig_down", "water", "ring")
CALM = np.array([a[0] not in UNASKED for a in ACTIONS])   # (what the learned body does unhurt and unsteered: no
                                                          # crafting, placing or digging of its own - trial 7: 22
                                                          # workbenches placed while the build waited; trial 5: its site dug)
PURSUE = next(k for k, a in enumerate(ACTIONS) if a[0] == "pursue")
NOT_MADE = ("dig", "till")                   # (the land dug, soil tilled: a tool's work, no material)
AS_ITEM = {"wall": "wood_wall", "floor": "plank", "stone_floor": "path"}              # (a wall of the pack's choosing: wood, the cheaper, counted)
_BUILT = {b.char for b in BLOCKS if b.name in set(PLACEABLE.values())}
LAND = {k for k, chars in DROPS.items() if chars - _BUILT}      # (what the land gives: a tree wood, a rock stone - not
                                                                # a wall dug back up)


def bill(demand: Counter, inv: dict) -> Counter:
    """The least raw material that makes `demand` from the pack `inv`. Demand is summed per item and expanded top
    down, an item only after every item that consumes it, so a recipe yielding several (a wood: 2 shingles) is
    crafted ceil(demand / yield) times, not once a cell; a station is made once (research notes, Theorem 1:
    exact for an acyclic recipe table with one recipe an item; expanded a cell at a time it counted 107 wood for
    107 shingles, 54 do)."""
    order, seen = [], set()

    def visit(k):                                     # (post-order: what an item is made of comes before it)
        if k in seen:
            return
        seen.add(k)
        r = RECIPES.get(k)
        if r is not None and k not in LAND:
            for c in [*r.inputs, *([r.station] if r.station else [])]:
                visit(c)
        order.append(k)
    for k in list(demand):
        visit(k)
    need, have, out, stations = Counter(demand), Counter(inv), Counter(), set()
    for k in reversed(order):                         # (consumers first: an item's whole demand is known)
        n = need[k] - have[k]
        r = RECIPES.get(k)
        if n <= 0:
            continue
        if k in LAND or r is None:
            out[k] += n
            continue
        times = -(-n // r.output.get(k, 1))
        for c, m in r.inputs.items():
            need[c] += m * times
        if r.station and r.station not in stations:
            stations.add(r.station)
            need[r.station] += 1
    return out


def lacking(plan, is_laid, inv: dict) -> Counter:
    """The raw material the plan's pieces not laid yet still take, beyond what the pack holds."""
    flights, demand = getattr(plan, "flights", {}), Counter()
    for cell in plan.cells:                  # (a flight of stairs is one piece of a stairs item a step)
        if cell[3] not in NOT_MADE and not is_laid(cell):
            demand[AS_ITEM.get(cell[3], cell[3])] += flights.get(cell[:2], 1) if cell[3] == "stairs" else 1
    return bill(demand, inv)


class Project:
    """The build under way: (action, goal) each decision while the plan stands unfinished, else None. Everything
    the unlaid pieces take is gathered first, in trips (the builder fetching a craft's worth at a time walked from
    the trees home and back for 1500 steps, the site out of view from one, the trees from the other), then laid."""

    BATCH = 24                               # (gathered in one go before going back to lay: a trip, not a walk a piece)

    def __init__(self):
        self.item, self.until = None, 0      # (the material being gathered, and how many to hold before laying again)

    def act(self, obs: dict, social, build_ok: bool):
        from .build import laid_by, plan_of
        if getattr(social, "plan", None) is None:
            return None
        inv = obs["self"].get("inventory", {})
        if self.item and inv.get(self.item, 0) < self.until:
            return self.gather(obs, social, build_ok)
        self.item = None
        plan = plan_of(social)
        need = lacking(plan, laid_by(obs, social, plan), inv) if getattr(social, "home", None) else Counter()
        need = Counter({k: n for k, n in need.items() if DROPS.get(k)})    # (earth, stone blocks: dug from the land,
        if not need or social.home is None:     # not gathered yet - #89; all in the pack: the builder lays it, as
            return (BUILD, None) if build_ok else None              # built() proves; no spot yet: it chooses one)
        self.item = max(need, key=lambda k: (need[k], k))
        self.until = inv.get(self.item, 0) + min(need[self.item], self.BATCH)
        return self.gather(obs, social, build_ok)

    def gather(self, obs, social, build_ok=False):
        """The gather verb after the item, or None when it has no way to it here (none in sight: the project waits);
        shut in a pocket, the build verb, which digs its way out first (rl/site.py way_out - live trial 12)."""
        from .actions import pursued, shut_in
        g = Goal("gather", self.item, None, obs.get("tick", 0), obs["self"].get("inventory", {}).get(self.item, 0))
        g.look = True                         # (none in view: it walks to look further - rl/pursue.py gather)
        if pursued(obs, g, social) is None:
            self.item = None
            return (BUILD, None) if build_ok and shut_in(obs, social) else None
        return PURSUE, g
