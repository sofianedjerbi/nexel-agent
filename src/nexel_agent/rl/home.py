"""The mind's side of a home (#30): where it is (the body chooses it, rl/build.py social.home; the mind keeps it in
memory.json across sessions and hands it back at the start: rl/mind.py), and one line of facts on it each plan: where
it is, what the judge says of it (rl/judge.py: v1 told in words, then what v2 and v3 find missing) and what it still
lacks of its plan (rl/build.py: the hut's blueprint, or the one the architect drew) beyond the pack, down to what that is made of (the game's recipes).
It changes nothing the body reads: no goal kind, no input.

The judge reads a world; the mind has the observation: Seen is the world as the observation shows it (its grid, its
legend, its heights and roofs, VIEW_R round the player). Past it nothing is known: open ground."""

import math
from collections import Counter

from nexel_worlds.sim.constants import DAY_TICKS, TICK_HZ, VIEW_R
from nexel_worlds.tiles import BLOCKS, GROUNDS, RECIPES

from . import judge
from .actions import LEVEL
from .build import HUT, WALLS, plan_of
from .facts import compass, counted

_BLOCK, _GROUND = {b.name for b in BLOCKS}, {g.name for g in GROUNDS}
NIGHT, SOON_S = .75, 300     # night: the day's last quarter (the game's is_night: glooms patrol); told from 5 min before
PIECE_ITEM = {"wall": "wood_wall", "door": "door", "roof": "roof"}     # (the blueprint's pieces as items; furniture is its own)


class Seen:
    """The observation as a world, for the judge: block, ground, roof and level of a tile in view."""

    def __init__(self, obs: dict):
        me, legend = obs["self"], obs.get("legend") or {}
        self.x0, self.y0 = me["x"] - VIEW_R, me["y"] - VIEW_R
        self.names = [[legend.get(ch, "") for ch in row] for row in obs.get("grid") or []]
        self.heights, self.roofs = obs.get("heights"), obs.get("roofs")

    def _at(self, rows, x, y):
        i, j = x - self.x0, y - self.y0
        return rows[j][i] if rows and 0 <= j < len(rows) and 0 <= i < len(rows[j]) else None

    def block(self, x, y):
        n = self._at(self.names, x, y)
        return n if n in _BLOCK else "none"

    def ground(self, x, y):
        n = self._at(self.names, x, y)
        return n if n in _GROUND else "grass"           # (under a block, or out of view: the meadow's)

    def roof(self, x, y):
        return int(self._at(self.roofs, x, y) == "1")

    def level(self, x, y):
        return LEVEL.get(self._at(self.heights, x, y) or "0", 0)


def stands(world, x, y, piece) -> bool:
    """Is this piece of the blueprint there (as judge.laid reads one: a roof over it, a wall, a door, the thing)?"""
    b = world.block(x, y)
    if piece == "path":
        return world.ground(x, y) == "cobble"
    return bool(world.roof(x, y)) if piece == "roof" else b in WALLS if piece == "wall" else \
        b in judge.ENCLOSE and not judge._SOLID.get(b, True) if piece == "door" else b == piece


def lacks(obs: dict, home, social=None) -> dict:
    """What the home still lacks, beyond the pack: the blueprint's pieces not standing, as items, those not carried
    made of what their recipe takes (one step down: wood, fiber, stone), all as {item: n}."""
    world, have = Seen(obs), dict((obs.get("self") or {}).get("inventory") or {})
    pieces: dict = {}
    plan = plan_of(social)
    if plan is not HUT and social is not None and getattr(social, "home", None):   # (a drawn build: each cell not done)
        from .build import laid_by
        is_laid = laid_by(obs, social, plan)
        left = [c[3] for c in plan.cells if not is_laid(c)]
        for p in left:
            if p == "dig" or (p == "till" and have.get("hoe")):
                continue                                     # (the land dug is the pack's; a hoe tills them all)
            item = "hoe" if p == "till" else PIECE_ITEM.get(p, p)
            pieces[item] = 1 if item == "hoe" else pieces.get(item, 0) + 1
    else:
        for dx, dy, p in plan.pieces:
            if not stands(world, home[0] + dx, home[1] + dy, p):
                item = PIECE_ITEM.get(p, p)
                pieces[item] = pieces.get(item, 0) + 1
    out: dict = {}
    for item, n in pieces.items():
        short, r = n - have.get(item, 0), RECIPES.get(item)
        if short <= 0:
            continue
        if r is None:
            out[item] = out.get(item, 0) + short
            continue
        times = math.ceil(short / r.output.get(item, 1))
        for k, m in r.inputs.items():
            out[k] = out.get(k, 0) + m * times
    return {k: n - have.get(k, 0) for k, n in out.items() if n > have.get(k, 0)}


def reading(obs: dict, home, social=None) -> dict:
    """What the judge and the blueprint say of the home, as facts (what told() words, and the brain view shows):
    {"at", "where", "far": out of sight, "verdict": v1 (rl/judge.py verdict, its report left out), "told": v1 in words,
    "sense", "looks": v2 and v3 ({"grade", "of", "failed"}, a closed home only), "lacks": {item: n}}."""
    me, world = obs["self"], Seen(obs)
    x, y = home
    r = {"at": [x, y], "where": compass(x - me["x"], y - me["y"]),
         "far": max(abs(x - me["x"]), abs(y - me["y"])) > VIEW_R - 2}
    if r["far"]:
        return r
    if plan_of(social) is not HUT:                           # (a build the architect drew: how far it is laid - a
        from .build import laid_by                           # dug room, a garden, the parts of a compound read on
        plan = plan_of(social)                               # the land's view are no room to judge)
        is_laid = laid_by(obs, social, plan)
        left = [c[3] for c in plan.cells if not is_laid(c)]
        r.update(pieces=[len(plan.cells) - len(left), len(plan.cells)], left=dict(Counter(left)),
                 lacks=lacks(obs, home, social))
        return r
    ix, iy = inside(home, social)
    v = judge.verdict(world, ix, iy)
    r.update(verdict={k: v[k] for k in ("grade", "of", "checks", "failed")}, told=judge.told(v), closed=v["report"]["closed"])
    if r["closed"]:
        for k, f in (("sense", judge.sense), ("looks", judge.looks)):
            s = f(world, ix, iy)
            r[k] = {"grade": s["grade"], "of": s["of"], "failed": s["failed"]}
    r["lacks"] = lacks(obs, home, social)
    return r


def told(obs: dict, home=None, r: dict | None = None) -> str:
    """The home in one line of facts: "Your home 12 tiles NW: closed, no roof (40%), no bed (grade 3 of 6); it makes
    no sense yet: no chest, no seat; it does not look good: no roof, no windows; it lacks 6 wood, 3 fiber". From its
    reading (r), or read now."""
    r = r or reading(obs, home)
    lacking = lambda k: f"some {k[3:].replace('_', ' ')}" if k.startswith("no_") else f"no {k.replace('_', ' ')}"   # (no_stray failed: "some stray", not "no no stray")
    if r["far"]:
        return f"Your home is {r['where']}: too far to see how it stands."
    if "pieces" in r:
        n, of = r["pieces"]
        left = ", ".join(f"{k} x{v}" if v > 1 else k for k, v in r["left"].items())
        return (f"Your build {r['where']}: {n} of {of} pieces done" + (f"; left: {left}" if left else ", whole")
                + (f"; it lacks {counted(r['lacks'])}." if r["lacks"] else "; you have all it takes."))
    line = f"Your home {r['where']}: {r['told']}"
    if r["closed"]:
        if r["sense"]["failed"]:
            line += "; it makes no sense yet: " + ", ".join(lacking(k) for k in r["sense"]["failed"])
        if r["looks"]["failed"]:
            line += "; it does not look good: " + ", ".join(lacking(k) for k in r["looks"]["failed"])
    return line + (f"; it lacks {counted(r['lacks'])}." if r["lacks"] else "; you have all it takes.")


def inside(home, social=None) -> tuple:
    """A tile inside the home, where the judge reads it from: the plan's first stand (the hut's middle, a drawn
    home's first floor tile; a drawn home's origin is its grid's corner, a wall)."""
    sx, sy = plan_of(social).ground[0]
    return home[0] + sx, home[1] + sy


def day(obs: dict, home, social=None) -> str | None:
    """A player's day (#15): from five minutes before night until dawn, one line: how long until night falls (or
    until dawn), where its home is, and its bed there (sleeping in a bed is where you wake after dying), or what it
    still needs for one. None in the day before that."""
    phase = (obs.get("tick", 0) % DAY_TICKS) / DAY_TICKS
    night = phase >= NIGHT
    left_s = ((1 if night else NIGHT) - phase) * DAY_TICKS / TICK_HZ
    if not night and left_s > SOON_S:
        return None
    mins = max(1, round(left_s / 60))
    when = f"It is night: glooms are out, dawn in {mins} min" if night else f"Night falls in {mins} min: glooms come out"
    if not home:
        return f"{when}; you have no home to go to: a bed is where you wake after dying."
    me = obs["self"]
    where = compass(home[0] - me["x"], home[1] - me["y"])
    world = Seen(obs)
    has = any(stands(world, home[0] + dx, home[1] + dy, p) for dx, dy, p in plan_of(social).pieces if p == "bed") or \
        judge.report(world, *inside(home, social)).get("contents", {}).get("bed", 0) > 0
    bed = "a bed" if has else "no bed yet: aim for one, and sleep in it"
    if max(abs(home[0] - me["x"]), abs(home[1] - me["y"])) > VIEW_R - 2:
        return f"{when}; your home is {where}: sleep in a bed there, it is where you wake after dying."
    return f"{when}; your home is {where}, with {bed} (a bed is where you wake after dying)."
