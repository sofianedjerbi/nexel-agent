"""What the body does when no planned command is ready: get out if stuck (escape.py), else the goal's next step
(filler.py). Every choice carries a routine name, and a routine the watchdog banned in this area is skipped:
the next one down the list is tried (watchdog.py)."""

import time

from .aim import HUNGRY, just_failed
from .escape import stuck, way_out
from .filler import filler, goal_absent
from .planner import Item
from .policy import DIRS
from .tiles import FOOD


def climbing(t, obs) -> Item | None:
    """A stair being cut goes on before any plan (a plan would pull the body off the wall)."""
    if t.memory.get("climb_to") is None or not stuck(obs, t.memory):
        return None
    return escape(t, obs)


def escape(t, obs) -> Item | None:
    me = obs["self"]
    banned = {b.split(":", 1)[1] for b in t.dog.banned(me["x"], me["y"], time.monotonic()) if b.startswith("escape:")}
    out = way_out(obs, t.memory, banned)
    return Item(*out[:2], "escape", out[2], routine=f"escape:{out[3]}") if out else None


def fallback(t, obs) -> Item:
    me = obs["self"]
    now = time.monotonic()
    banned = t.dog.banned(me["x"], me["y"], now)
    if stuck(obs, t.memory) and (it := escape(t, obs)):
        return it                                 # no walk works from here: get out first
    hungry = me["hunger"] <= HUNGRY and not any(me["inventory"].get(f) for f in FOOD)
    goal = "food" if hungry else t.goal
    anchor = t.anchor[:2] if t.anchor and now < t.anchor[2] else None
    if goal_absent(goal, obs):                    # what the goal wants is nowhere in view: no spot to hold, go find it
        anchor, goal = None, f"find {goal}"
    routine = "filler"
    if "filler" in banned:                        # its usual step did nothing here for a while: go somewhere else
        anchor, goal, routine = None, "explore", "filler:explore"
    f = None if routine in banned else filler(obs, goal, t.memory, t.rng, anchor)
    if f and just_failed(t.memory, f[0], f[1], now):
        t.stats["backoff"] += 1                   # that just failed: not again within a second
        f = None
    if f is None and (it := escape(t, obs)):
        return it
    f = f or ("look_far", {"dir": t.rng.choice(list(DIRS))}, "looking around")
    return Item(*f[:2], "filler", f[2], routine=routine)
