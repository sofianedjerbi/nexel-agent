"""A build of storeys judged (#66): each storey by the judge's own rules (rl/judge.py: v1, v2, v3) on a view of that
storey (StoreyView: what stands on its floor, a floor tile walkable, a tile covered when a block is over it), and the
whole by what its kind is for. A home (rect or round, roofed): every storey closed with a door and covered, a bed, a
light, a chest and a seat somewhere, all reached; a base (no roof): a closed yard with a gate, a fire in it.

    judged(spec) -> {"ok", "grade", "of", "failed", "told", "structure"}"""

from nexel_worlds.sim.storeys import STOREY

from . import judge
from .structure import Structure

LIGHT = ("campfire", "torch")


class StoreyView:
    """Storey k of a build in a sim world, as the judge reads a world: block, ground, roof, level at a tile."""

    def __init__(self, sim, z):
        self.sim, self.z = sim, z

    def block(self, x, y):
        c = self.sim.floors.get(x, y, self.z)
        return c[1] if c else "none"

    def ground(self, x, y):
        return "grass" if self.sim.floors.get(x, y, self.z) else "water"      # (no floor: nothing to stand on)

    def roof(self, x, y):
        return int(any(self.sim.world.solid(x, y, z) for z in range(self.z + 1, self.z + STOREY + 1)))

    def level(self, x, y):
        return self.z


def stamp(sim, s: Structure, x0, y0) -> int:
    """The build at (x0, y0) (its walls' top-left) in `sim`, every cell laid, on flat grass cleared round it; its
    land's level."""
    lv = sim.world.level(x0, y0)
    ts = s.clear()
    for x in range(min(a for a, _ in ts) - 3, max(a for a, _ in ts) + 4):
        for y in range(min(b for _, b in ts) - 3, max(b for _, b in ts) + 4):
            sim.world.set(x0 + x, y0 + y, ground="grass", block="none", level=lv)
    for dx, dy, dz, p in s.cells():
        X, Y = x0 + dx, y0 + dy
        if p == "dig":
            sim.world.set_cell(X, Y, lv + dz, False)
            continue
        if p in ("till", "seed"):                         # (soil tilled; a seed sown is a sprout)
            sim.world.set(X, Y, **({"ground": "tilled"} if p == "till" else {"block": "sprout"}))
            continue
        if dz < 0:                                        # (on a dug floor: the tile's lowest ground)
            sim.world.set(X, Y, block=p)
            continue
        if p == "path":
            sim.world.set(X, Y, ground="cobble")
        elif p == "stairs":
            continue                                      # (the way up is the structure's: s.stairs)
        elif p in ("plank", "roof"):
            sim.set_block(X, Y, lv + dz, p, thing=(sim.floors.get(X, Y, lv + dz) or (None, "none"))[1])
        elif p in ("stone_block", "dirt"):
            sim.world.set_cell(X, Y, lv + dz, True)          # (land in a cell: put_land's)
        elif dz == 0:
            sim.world.set(X, Y, block=judge.ENCLOSE[1] if p == "wall" else p)
        else:
            kind = (sim.floors.get(X, Y, lv + dz) or ("plank",))[0]
            sim.set_block(X, Y, lv + dz, kind, thing=judge.ENCLOSE[1] if p == "wall" else p)
    return lv


def judged(spec) -> dict:
    """The design judged, or what is wrong with it told."""
    from .core import SimEnv
    from .dug import design_of
    try:
        s = design_of(spec)
    except (KeyError, TypeError, ValueError) as e:
        return {"ok": False, "error": str(e), "grade": 0, "of": 1, "failed": [], "told": str(e)}
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    x0, y0 = a.x - 24, a.y - 24
    return {**judge_at(sim, s, x0, y0, stamp(sim, s, x0, y0)), "structure": s}


def judge_at(sim, s: Structure, x0, y0, lv) -> dict:
    """The build `s` as it stands in `sim` at (x0, y0) (its walls' top-left), its land at level `lv`, judged."""
    if hasattr(s, "beds"):
        return garden_at(sim, s, x0, y0)
    from .dug import DEPTH, DugView
    inside = sorted(s.inside)
    dug = hasattr(s, "digs")
    views = [DugView(sim, lv - DEPTH)] if dug else [sim.world if k == 0 else StoreyView(sim, lv + STOREY * k) for k in range(s.n)]
    per = []
    for k, v in enumerate(views):
        fx, fy = next(p for p in inside if p not in s.storeys[k])            # (a free floor tile: the judge reads from it)
        px, py = x0 + fx, y0 + fy
        per.append({"v1": judge.verdict(v, px, py), "v2": judge.sense(v, px, py), "v3": judge.looks(v, px, py)})
    things = [t for st in s.storeys for t in st.values()]
    base = not s.roofed and not dug
    checks = {}
    for k, r in enumerate(per):
        tag = f"storey {k}: " if s.n > 1 else ""
        checks[tag + "closed"] = r["v1"]["checks"]["closed"]
        checks[tag + "door"] = r["v1"]["checks"]["door"]
        checks[tag + "reachable"] = r["v2"]["checks"]["reachable"]
        if not base:
            checks[tag + "roofed"] = r["v1"]["checks"]["roofed"]       # (dug: the land over it)
    g = per[0]
    checks["room_enough"] = g["v1"]["checks"]["room_enough"]
    checks["door_out"] = g["v2"]["checks"]["door_out"]
    checks["lit"] = any(t in LIGHT for t in things)
    if not base:
        checks.update(bed="bed" in things, chest="chest" in things, seat=any(t in ("chair", "table") for t in things),
                      fit=g["v2"]["checks"]["fit"] or s.n > 1, light_by_bed=all(r["v2"]["checks"]["light_by_bed"] for r in per
                                                                                if "bed" in (r["v2"]["facts"].get("core") or {})))
        if not dug:
            checks["roof_over"] = per[-1]["v3"]["checks"]["roof"]
    if not dug:                                           # (a dug room's looks are the land's: its trench is its way)
        checks["path"] = g["v3"]["checks"]["path"]
        checks["no_stray"] = g["v3"]["checks"]["no_stray"]
    if s.shape == "rect" and not dug:
        checks["rectangles"] = g["v3"]["checks"]["rectangles"]
        checks["straight"] = g["v3"]["checks"]["straight"]
    if not dug:
        checks["symmetric"] = g["v3"]["checks"]["symmetric"]
    failed = [k for k, ok in checks.items() if not ok]
    return {"ok": True, "grade": len(checks) - len(failed), "of": len(checks), "failed": failed, "checks": checks,
            "told": "; ".join(failed) or "nothing failed", "kind": "dug" if dug else "base" if base else "home",
            "storeys": [r and {k: r[k]["failed"] for k in r} for r in per]}


WHY = {"roofed": "not covered (a floor or the roof over it)", "lit": "no light (a campfire)", "bed": "no bed",
       "chest": "no chest", "seat": "no table or chair", "roof_over": "the roof stops at the walls",
       "closed": "the walls do not close it", "door": "no door in its walls",
       "reachable": "a piece is not reached on foot from its door (furniture blocks the way)"}


def told(j) -> str:
    """The failed checks in words, as the architect is told them."""
    from . import blueprint as bp
    out = []
    for f in j["failed"]:
        where, _, name = f.rpartition(": ")
        out.append((where + ": " if where else "") + WHY.get(name, bp.WHY.get(name, name)))
    return "; ".join(out) or "nothing failed"


def built(spec, seed=5, picks=800) -> dict:
    """The design laid by the build verb alone in a sim world, from a pack holding every piece, and judged as it
    stands: {"grade", "of", "failed", "picks"}, or {"sited": False} when no spot in view fits it."""
    from collections import Counter

    from .actions import ACTIONS
    from .core import SimEnv
    from .plan import of_structure
    from .dug import design_of
    s = design_of(spec)
    kit = Counter("stone_wall" if p == "wall" else p for *_, p in s.cells() if p != "dig")
    kit["pickaxe"] = 1
    kit.pop("till", None)
    kit["hoe"] = 1
    kit["stairs"] = sum(len(steps) for _, steps, _, _ in s.stairs)
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(seed)
    env.folk = []
    env.social.plan = of_structure(s)
    env.a.inventory.clear()
    env.a.inventory.update(kit)
    env.obs = env.sim.observe(env.a)
    b = next(i for i, a in enumerate(ACTIONS) if a[0] == "build")
    n = 0
    for n in range(picks):
        if not env.encoded()["mask"][b]:
            break
        env.step(b)
    if env.social.home is None:
        return {"sited": False}
    hx, hy = env.social.home
    lv = env.sim.world.level(hx + min(s.inside)[0], hy + min(s.inside)[1])
    if hasattr(s, "digs"):                                # (dug: the land's top, its level before the digging)
        lv = env.sim.world.level(hx + s.top[0], hy + s.top[1])
    j = judge_at(env.sim, s, hx, hy, lv)
    return {"sited": True, "picks": n, **{k: j[k] for k in ("grade", "of", "failed")}}


def garden_at(sim, s, x0, y0) -> dict:
    """A garden as it stands: fenced all round (a fence closes a plot, if not a room: judge.LOOK_CLOSERS), a gate,
    its beds tilled and sown (nine in ten at least), a path out of the gate."""
    w = sim.world
    walk = next(iter(s.walks or s.beds))
    tiles = judge.inside(w, x0 + walk[0], y0 + walk[1], judge.LOOK_CLOSERS)
    beds = [(x0 + x, y0 + y) for x, y in s.beds]
    gate = (x0 + s.doors[0][0], y0 + s.doors[0][1])
    tilled = sum(w.ground(*p) in ("tilled", "wet_tilled") for p in beds) / max(1, len(beds))
    sown = sum(w.block(*p) in ("sprout", "crop", "ripe_crop") for p in beds) / max(1, len(beds))
    checks = {"fenced": tiles is not None, "gate": w.block(*gate) == "gate", "tilled": tilled >= .9,
              "sown": sown >= .9, "path": judge._path(w, gate, (tiles or set()) | {gate}) >= 2 if tiles else False,
              "room_to_grow": len(beds) >= 4}
    failed = [k for k, ok in checks.items() if not ok]
    return {"ok": True, "grade": len(checks) - len(failed), "of": len(checks), "failed": failed, "checks": checks,
            "told": "; ".join(failed) or "nothing failed", "kind": "garden"}
