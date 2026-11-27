"""A build is a project the body works at to its end (#87). The model plans, code executes it (Voyager, Mindcraft,
CraftUtopia: the model names what, skills run to the end): each decision, what the pack allows is laid (the build
verb, rl/build.py), else the raw material the unlaid pieces lack most is gathered (the gather verb, rl/pursue.py),
read from the game's own recipes down to what the land gives. Live trials 4 to 6 built nothing drawn: the builder
stopped when the pack was short, and gathering the hundreds of pieces a house takes was left to the mind's goals,
set every 20 s, with the learned body (paid for crafting every thing once) acting between them."""

from collections import Counter

from nexel_worlds.tiles import BLOCKS, PLACEABLE, RECIPES

from .actions import ACTIONS
from .goals import Goal
from .pursue import DROPS

BUILD = next(k for k, a in enumerate(ACTIONS) if a[0] == "build")
PURSUE = next(k for k, a in enumerate(ACTIONS) if a[0] == "pursue")
NOT_MADE = ("dig", "till")                   # (the land dug, soil tilled: a tool's work, no material)
AS_ITEM = {"wall": "wood_wall"}              # (a wall of the pack's choosing: wood, the cheaper, counted)
_BUILT = {b.char for b in BLOCKS if b.name in set(PLACEABLE.values())}
LAND = {k for k, chars in DROPS.items() if chars - _BUILT}      # (what the land gives: a tree wood, a rock stone - not
                                                                # a wall dug back up)


def raw(item: str, n: int, have: Counter, out: Counter, depth: int = 0):
    """What `n` of `item` takes from the land, what `have` holds used first (and taken from it), added to `out`."""
    use = min(have[item], n)
    have[item] -= use
    n -= use
    r = RECIPES.get(item)
    if n <= 0:
        return
    if item in LAND or r is None or depth > 4:
        out[item] += n
        return
    if r.station and not have[r.station]:                       # (a station made once, then used for every craft)
        raw(r.station, 1, have, out, depth + 1)
        have[r.station] += 1
    times = -(-n // r.output.get(item, 1))
    for k, m in r.inputs.items():
        raw(k, m * times, have, out, depth + 1)


def lacking(plan, is_laid, inv: dict) -> Counter:
    """The raw material the plan's pieces not laid yet still take, beyond what the pack holds."""
    have, out = Counter(inv), Counter()
    for cell in plan.cells:
        if cell[3] not in NOT_MADE and not is_laid(cell):
            raw(AS_ITEM.get(cell[3], cell[3]), 1, have, out)
    return out


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
            return self.gather(obs, social)
        self.item = None
        plan = plan_of(social)
        need = lacking(plan, laid_by(obs, social, plan), inv) if getattr(social, "home", None) else Counter()
        need = Counter({k: n for k, n in need.items() if DROPS.get(k)})    # (earth, stone blocks: dug from the land,
        if not need or social.home is None:     # not gathered yet - #89; all in the pack: the builder lays it, as
            return (BUILD, None) if build_ok else None              # built() proves; no spot yet: it chooses one)
        self.item = max(need, key=lambda k: (need[k], k))
        self.until = inv.get(self.item, 0) + min(need[self.item], self.BATCH)
        return self.gather(obs, social)

    def gather(self, obs, social):
        """The gather verb after the item, or None when it has no way to it here (none in sight: the project waits)."""
        from .actions import pursued
        g = Goal("gather", self.item, None, obs.get("tick", 0), obs["self"].get("inventory", {}).get(self.item, 0))
        g.look = True                         # (none in view: it walks to look further - rl/pursue.py gather)
        if pursued(obs, g, social) is None:
            self.item = None
            return None
        return PURSUE, g
