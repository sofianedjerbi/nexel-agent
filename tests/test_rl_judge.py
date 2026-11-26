"""The build judge (rl/judge.py, v1): the game's reading of a build as checks a player would name."""

from nexel_agent.rl import judge
from nexel_agent.rl.core import SimEnv


def ring(sim, x, y, w, h, door=None):
    for dx in range(-1, w + 1):
        for dy in range(-1, h + 1):
            if dx in (-1, w) or dy in (-1, h):
                sim.world.set(x + dx, y + dy, block="door" if (x + dx, y + dy) == door else "stone_wall")


def test_a_room_grows_check_by_check():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    assert judge.verdict(sim.world, a.x, a.y)["grade"] == 0
    x, y = a.x - 1, a.y - 1
    ring(sim, x, y, 3, 3)
    v = judge.verdict(sim.world, a.x, a.y)
    assert v["checks"]["closed"] and v["checks"]["room_enough"] and not v["checks"]["door"] and v["grade"] == 2
    ring(sim, x, y, 3, 3, door=(x + 1, y - 1))
    sim.world.set(x + 2, y + 2, block="bed")
    sim.world.set(x, y + 2, block="campfire")
    v = judge.verdict(sim.world, a.x, a.y)
    assert v["failed"] == ["roofed"] and v["grade"] == 5
    assert judge.told(v).startswith("closed, a door, no roof (0% covered), lit, a bed")


def test_the_build_potential_pays_as_it_grows_and_back_when_torn_down():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    p, ok = judge.Potential(), [("place", {}, {"ok": True})]
    x, y = a.x - 1, a.y - 1
    ring(sim, x, y, 3, 3)
    assert p.step(sim.world, None, (a.x, a.y), ok) == 2 * judge.Potential.UNIT        # (closed, room enough)
    assert p.step(sim.world, None, (a.x + 9, a.y), ok) == 0                              # (walked out: no loss)
    sim.world.set(x + 1, y - 1, block="none")
    assert p.step(sim.world, None, (a.x + 9, a.y), [("dig", {}, {"ok": True})]) == -2 * judge.Potential.UNIT


def test_each_piece_of_the_home_laid_pays():
    from nexel_agent.rl.build import PLAN
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    home = (a.x + 3, a.y + 3)
    p, ok = judge.Potential(), [("place", {}, {"ok": True})]
    assert p.step(sim.world, home, (a.x, a.y), ok) == 0
    dx, dy, _ = PLAN[0]
    sim.world.set(home[0] + dx, home[1] + dy, block="stone_wall")
    assert abs(p.step(sim.world, home, (a.x, a.y), ok) - judge.Potential.PIECE) < 1e-9
    assert judge.laid(sim.world, home) == 1


# ---- v2 (#31): does it make sense? One good home, then one change each, each failing its own check

def home(long=3, change=None):
    """A good home for one: 3 by `long` inside, its door in the north wall's middle onto grass, a campfire (its light
    and station), a chair, a chest and a bed, each beside a walkable tile. `change`: {(dx, dy): block}, from its corner."""
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    x, y = a.x - 1, a.y - 1
    for i in range(-4, 8):
        for j in range(-4, long + 4):
            sim.world.set(x + i, y + j, ground="grass", block="none")
    ring(sim, x, y, 3, long, door=(x + 1, y - 1))
    for (i, j), b in {(2, 0): "campfire", (0, 0): "chair", (0, 2): "chest", (2, 2): "bed", **(change or {})}.items():
        if b == "dirt":                                                            # (a block of earth put down: the land one up)
            sim.world.set(x + i, y + j, level=sim.world.level(x + i, y + j) + 1)
        else:
            sim.world.set(x + i, y + j, block=b)
    return judge.sense(sim.world, a.x, a.y), (sim, x, y)


def test_a_good_home_makes_sense():
    v, _ = home()
    assert v["failed"] == [] and v["grade"] == v["of"] == len(judge.SENSE), v
    assert v["facts"]["core"] == {"campfire": 1, "chair": 1, "chest": 1, "bed": 1} and v["facts"]["area"] == 9


def test_each_thing_a_home_lacks_is_named():
    for change, failed in (({(0, 2): "none"}, ["chest"]),                          # no chest
                           ({(2, 0): "fire"}, ["station"]),                        # a light, nothing to make things at
                           ({(0, 0): "none"}, ["seat"]),                           # nowhere to sit
                           ({(2, 2): "none", (1, 0): "bed"}, ["reachable"]),       # the bed in the doorway: in, nothing is reached
                           ({(1, 2): "dirt"}, ["tidy"]),                           # a block of earth left in the room
                           ({(2, 0): "workbench"}, ["light_by_bed"])):             # a station, no light by the bed
        v, _ = home(change=change)
        assert v["failed"] == failed, (change, v["failed"], v["facts"])


def test_a_door_onto_water_and_a_hall_do_not_make_sense():
    _, (sim, x, y) = home()
    sim.world.set(x + 1, y - 2, ground="water")                                    # the door opens onto the pond
    v = judge.sense(sim.world, x + 1, y + 1)
    assert v["failed"] == ["door_out"] and v["facts"]["shut_doors"] == [(x + 1, y - 1)]
    v, _ = home(long=14)                                                           # 3 by 14: 42 tiles, a hall
    assert v["failed"] == ["fit"] and v["facts"]["area"] == 42
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    v = judge.sense(env.sim.world, env.a.x, env.a.y)                               # open land: nothing to judge
    assert v["grade"] == 0 and v["failed"] == list(judge.SENSE) and not v["facts"]["closed"]


# ---- v3 part 1 (#32): does it look good? A builder's good hut, then one change each

def hut(change=None, overhang=True, path=True):
    """3 by 3 inside, wood walls and corners, the door in the north wall's middle, a window (a fence) in the east and
    west walls, a roof over it all and one tile past, a cobble path of three north of the door. `change`: {(dx, dy):
    block} from the inside's corner."""
    from nexel_worlds.sim.storeys import STOREY
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    x, y = a.x - 1, a.y - 1
    for i in range(-6, 9):
        for j in range(-7, 9):
            sim.world.set(x + i, y + j, ground="grass", block="none")
    lv = sim.world.level(x, y)
    for i in range(-1, 4):
        for j in range(-1, 4):
            if i in (-1, 3) or j in (-1, 3):
                sim.world.set(x + i, y + j, block="wood_wall")
    pieces = {(1, -1): "door", (-1, 1): "fence", (3, 1): "fence", **(change or {})}
    for (i, j), b in pieces.items():
        sim.world.set(x + i, y + j, block=b)
    r = 2 if overhang else 1
    for i in range(-r, 3 + r):
        for j in range(-r, 3 + r):
            sim.set_block(x + i, y + j, lv + STOREY, "roof")
    for j in range(-4, -1) if path else ():
        for i in (0, 1, 2):
            sim.world.set(x + i, y + j, ground="cobble")
    return judge.looks(sim.world, x + 1, y + 2)                                  # (from the south middle: the middle may be a post)


def test_a_builders_hut_looks_good():
    v = hut()
    assert v["failed"] == [] and v["grade"] == v["of"] == len(judge.LOOKS), v
    assert v["facts"]["materials"] == {"wood_wall": 13} and v["facts"]["windows"] and v["facts"]["path"] == 9
    v = hut({(-1, -1): "stone_wall", (3, -1): "stone_wall", (-1, 3): "stone_wall", (3, 3): "stone_wall"})
    assert v["checks"]["material"], v["facts"]                                    # a second material, at the corners: on purpose


def test_each_thing_that_spoils_a_hut_is_named():
    for kw, failed in ((dict(change={(0, 3): "stone_wall", (2, 3): "stone_wall"}), ["material"]),   # two of another, here and there
                       (dict(change={(1, 1): "wood_wall"}), ["rectangles"]),                           # a post in the middle
                       (dict(change={(0, 0): "wood_wall", (2, 0): "wood_wall", (-1, -1): "none", (3, -1): "none"}), ["straight"]),   # one-tile jogs: the north corners notched
                       (dict(overhang=False), ["roof"]),                                               # a roof that stops at the walls
                       (dict(change={(1, -1): "wood_wall", (0, -1): "door"}), ["symmetric"]),          # the door off its middle
                       (dict(change={(-1, 1): "wood_wall", (3, 1): "wood_wall"}), ["windows"]),        # no window
                       (dict(path=False), ["path"]),                                                   # no path to the door
                       (dict(change={(5, 4): "fence"}), ["no_stray"])):                                # a fence post left in the grass
        v = hut(**kw)
        assert v["failed"] == failed, (kw, v["failed"], v["facts"])


def test_a_home_begun_pays_only_for_what_it_adds():
    import random
    from nexel_agent.rl import starts
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, home_pct=100)
    for seed in range(8):
        env.reset(seed)
        if env.social.home:
            break
    assert env.social.home
    phi0 = env.build_phi.phi
    assert phi0 > 0
    r = env.build_phi.step(env.sim.world, env.social.home, (env.a.x, env.a.y), [("dig", {}, {"ok": True})])
    assert abs(r) < 1e-9                                  # (nothing changed: nothing paid for the walls it found)
