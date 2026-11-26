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
