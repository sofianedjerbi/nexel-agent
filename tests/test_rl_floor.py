"""A home's floor (the owner: "agent houses are missing a floor, they keep the default floor"): the ground storey's
inside laid over, planks a wooden floor, a path a stone one (rl/floor.py); a home left on the bare land is told."""

from test_rl_project import HOUSE, run

from nexel_agent.rl import judge3d
from nexel_agent.rl.build import plan_of


def _inside_grounds(env):
    hx, hy = env.social.home
    return {env.sim.world.ground(hx + x, hy + y) for x, y, z, p in plan_of(env.social).cells if p in ("floor", "stone_floor")}


def test_a_house_is_laid_with_a_wooden_floor_and_a_stone_one_when_chosen():
    env, left, n = run(HOUSE, 9, 3000)
    assert not left, (n, left[:4])
    assert _inside_grounds(env) == {"floor"}
    env, left, n = run({**HOUSE, "floor": "path"}, 9, 3000)
    assert not left, (n, left[:4])
    assert _inside_grounds(env) == {"cobble"}


def test_a_home_on_the_bare_land_is_told_to_lay_a_floor_and_a_yard_is_not():
    j = judge3d.judged({**HOUSE, "floor": False})
    assert "floored" in j["failed"] and "lay a floor" in judge3d.told(j)
    assert "floored" not in judge3d.judged(HOUSE)["failed"]
    assert "floored" not in judge3d.judged({"shape": "round", "radius": 4, "door": "south", "roof": False,
                                            "furniture": {"campfire": [3, 3]}})["failed"]
    assert "a floor is" in judge3d.judged({**HOUSE, "floor": "marble"})["error"]
