"""The `build` verb (rl/build.py): picked again and again in the game's own sim, it lays a hut a player would call a
home - closed, a door, a fire, a bed, a roof - and the judge (rl/judge.py) says so."""

from nexel_agent.rl import judge
from nexel_agent.rl.build import W, H
from nexel_agent.rl.core import SimEnv
from rl_kit import idx

BUILD = idx("build")


def test_a_hut_from_the_pack_by_building_alone():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    env.folk = []
    env.a.inventory.clear()
    env.a.inventory.update({"stone_wall": 16, "door": 1, "bed": 1, "campfire": 1, "roof": 25})
    env.obs = env.sim.observe(env.a)
    for _ in range(400):
        o = env.encoded()
        if not o["mask"][BUILD]:
            break
        env.step(BUILD)
    if "roofs" not in env.sim.observe(env.a):
        import pytest
        pytest.skip("the game's observation shows no roofs yet (sofianedjerbi/nexel-worlds#45)")
    hx, hy = env.social.home
    v = judge.verdict(env.sim.world, hx + W // 2, hy + H // 2)
    assert v["grade"] == v["of"], judge.told(v)


def test_a_start_with_a_home_kit_holds_the_hut_pieces():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, home_pct=100)
    env.reset(7)
    inv = env.a.inventory
    assert inv.get("door", 0) + inv.get("roof", 0) + inv.get("stone_wall", 0) + inv.get("wood_wall", 0) > 10


def test_a_pit_dug_in_the_spot_makes_it_choose_another():
    """The body dug a pit where its hut was to stand: the game refuses walls beside lower ground, so the spot is left."""
    from nexel_agent.rl.actions import built
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    env.folk = []
    env.a.inventory.update({"stone_wall": 16})
    o = env.sim.observe(env.a)
    built(o, env.social)
    hx, hy = first = env.social.home
    w = env.sim.world
    w.set(hx + 1, hy + 1, level=w.level(hx + 1, hy + 1) - 1)
    built(env.sim.observe(env.a), env.social)
    assert env.social.home != first


def test_a_home_at_the_edge_of_view_is_kept():
    """Ten tiles off, part of the ring round the spot is out of view: unknown is not lower, the home stays."""
    from nexel_agent.rl.actions import built
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    env.folk = []
    env.a.inventory.update({"stone_wall": 16})
    built(env.sim.observe(env.a), env.social)
    home = env.social.home
    env.a.x, env.a.y = home[0] + 1 - 9, home[1] + 1
    built(env.sim.observe(env.a), env.social)
    assert env.social.home == home


def test_no_roof_goes_on_an_open_ring():
    """Short of walls (spent elsewhere), it lays what walls it has and no roof: a roof on an open ring is lost."""
    from nexel_agent.rl.actions import built
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    env.folk = []
    env.a.inventory.clear()
    env.a.inventory.update({"stone_wall": 8, "door": 1, "roof": 25})
    cmd = built(env.sim.observe(env.a), env.social)
    assert cmd[0] == "ring" and not any(it == "roof" for _, _, it in cmd[1]["pieces"])


def test_a_home_begun_is_one_build_from_done():
    from nexel_agent.rl import starts
    import random
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    env.folk = []
    env.a.inventory.clear()
    home = starts.with_a_home_begun(env.sim, env.a, random.Random(1))
    env.social.home = home
    env.obs = env.sim.observe(env.a)
    v = judge.verdict(env.sim.world, home[0] + 1, home[1] + 1)
    assert v["checks"]["closed"] and v["checks"]["door"]
    env.step(BUILD)
    assert judge.verdict(env.sim.world, home[0] + 1, home[1] + 1)["grade"] == 6
