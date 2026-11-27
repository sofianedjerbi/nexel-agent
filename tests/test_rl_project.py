"""A build is a project the body works at to its end (#87): from an empty pack, the design drawn stands through the
project alone - the material gathered from the land as the game's recipes say, made, laid."""

from types import SimpleNamespace

import pytest

from nexel_agent.rl.build import laid_by, plan_of
from nexel_agent.rl.core import SimEnv
from nexel_agent.rl.dug import design_of
from nexel_agent.rl.plan import of_structure
from nexel_agent.rl.project import BUILD, Project, lacking

HOUSE = {"width": 3, "depth": 3, "door": "south", "roof": {"style": "gable"},
         "furniture": {"bed": [0, 0], "campfire": [2, 0], "chest": [0, 2], "chair": [2, 2]}}


def run(spec, seed, steps):
    env = SimEnv(ep_ticks=10 ** 7, roam=0, rivals=0)
    env.reset(seed)
    env.folk = []
    env.goals = SimpleNamespace(goal=None, step=lambda *a: 0, draw=lambda *a, **k: None, counts={})
    env.social.plan = of_structure(design_of(spec))
    env.a.inventory.clear()
    p, n = Project(), 0
    for n in range(steps):
        a = p.act(env.obs, env.social, bool(env.encoded()["mask"][BUILD]))
        if a is None:
            break
        env.goals.goal = a[1]
        env.step(a[0])
    plan = plan_of(env.social)
    left = [c for c in plan.cells if not laid_by(env.obs, env.social, plan)(c)] if env.social.home else plan.cells
    return env, left, n


def test_what_a_plan_lacks_is_read_down_to_the_land():
    plan = of_structure(design_of(HOUSE))
    need = lacking(plan, lambda c: False, {})
    assert set(need) <= {"wood", "stone", "fiber"} and need["wood"] > 50
    assert lacking(plan, lambda c: True, {}) == {}


@pytest.mark.parametrize("seed", [7])
def test_from_an_empty_pack_a_house_stands_through_the_project_alone(seed):
    env, left, n = run(HOUSE, seed, 6000)
    assert env.social.home is not None and left == [], (n, len(left), left[:6], dict(env.a.inventory))
