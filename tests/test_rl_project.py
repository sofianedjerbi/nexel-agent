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


def test_the_bill_is_the_least_raw_material_a_demand_takes():
    """Research notes, Theorem 1. A recipe yielding two (a wood: 2 shingles) is crafted ceil(n / 2) times for n,
    not once a cell (107 shingles: 54 wood, not 107); a station once; what the pack holds first."""
    from collections import Counter

    from nexel_agent.rl.project import bill
    assert bill(Counter({"roof": 107}), {}) == {"wood": 54}
    assert bill(Counter({"wood_wall": 4}), {}) == {"wood": 6 + 4}          # (2 crafts of 3 wood, a workbench of 4)
    assert bill(Counter({"wood_wall": 4}), {"workbench": 1, "wood": 2}) == {"wood": 4}
    assert bill(Counter({"roof": 3, "plank": 3}), {"roof": 1}) == {"wood": 1 + 2}


def test_what_a_plan_lacks_is_read_down_to_the_land():
    plan = of_structure(design_of(HOUSE))
    need = lacking(plan, lambda c: False, {})
    assert set(need) <= {"wood", "stone", "fiber"} and need["wood"] > 50
    assert lacking(plan, lambda c: True, {}) == {}


@pytest.mark.parametrize("seed", [7])
def test_from_an_empty_pack_a_house_stands_through_the_project_alone(seed):
    env, left, n = run(HOUSE, seed, 6000)
    assert env.social.home is not None and left == [], (n, len(left), left[:6], dict(env.a.inventory))


def test_the_build_the_mind_names_is_kept_whatever_its_goal_till_another_or_none(tmp_path):
    """Live trial 6: the mind named "a 2-storey round house with gabled shingle roof" with a gather goal, twice, and
    both were dropped (the words were kept only with the goal build)."""
    from nexel_agent.rl.mind import Mind
    m = Mind("Rook", "", "http://x", None, tmp_path / "m.json")
    m.apply({"goal": "gather", "item": "fiber", "build": "a round house with a gabled shingle roof"})
    assert m.build_words == "a round house with a gabled shingle roof" and m.goal.kind == "gather"
    m.apply({"goal": "craft", "item": "axe", "build": ""})
    assert m.build_words == "a round house with a gabled shingle roof"
    m.apply({"goal": "gather", "item": "wood", "build": "none"})
    assert m.build_words is None


def test_a_flight_of_stairs_is_counted_a_stairs_item_a_step():
    from test_rl_structure import DESIGNS
    s = design_of(DESIGNS["tower"])
    plan = of_structure(s)
    steps = sum(len(st) for _, st, _, _ in s.stairs)
    only_stairs = lambda c: c[3] != "stairs"                                 # noqa: E731  (all else laid)
    assert lacking(plan, only_stairs, {}) == {"wood": -(-steps // 2)}       # (a wood makes 2 stairs)




def test_a_plank_laid_up_a_storey_is_not_read_as_the_wall_standing_on_it():
    """The tower (#91): kept by its place alone, the first floor's plank read as the wall on it laid; the wall never
    went up, the second flight ran to the first floor and the top storey was never reached."""
    from nexel_agent.rl.social import SocialMemory
    s = SocialMemory("Rook", ("wood",))
    s.achieved("place", {"item": "plank", "x": 4, "y": 2, "z": 3}, True, 0, 99)
    assert (4, 2, 3, "block") in s.up_laid and (4, 2, 3, "thing") not in s.up_laid
    s.achieved("place", {"item": "wood_wall", "x": 4, "y": 2, "z": 3}, True, 0, 99)
    assert (4, 2, 3, "thing") in s.up_laid


def test_a_flight_waits_for_its_doors_column():
    from test_rl_structure import DESIGNS
    plan = of_structure(design_of(DESIGNS["tower"]))
    top = [c for c in plan.cells if c[3] == "stairs"][-1]
    after = plan.after[top]
    assert any(c[3] == "door" and c[2] == 6 for c in after) and any(c[3] == "wall" and c[2] == 3 for c in after)
    assert not plan.held(top, lambda c: c not in after) and plan.held(top, lambda c: True)


@pytest.mark.parametrize("seed", [9])
def test_from_an_empty_pack_a_tower_of_three_storeys_stands(seed):
    from test_rl_structure import DESIGNS
    env, left, n = run(DESIGNS["tower"], seed, 2000)
    assert left == [], (n, len(left), left[:4])


def test_unhurt_and_unsteered_the_body_moves_eats_and_talks_but_makes_nothing_of_its_own():
    """Trial 7: with the build waiting, the learned body placed 22 workbenches; trial 5: it dug its own site."""
    from nexel_agent.rl.actions import ACTIONS
    from nexel_agent.rl.project import CALM
    kinds = {a[0] for a, ok in zip(ACTIONS, CALM, strict=True) if ok}
    assert not kinds & {"craft", "place", "dig", "dig_at", "dig_down"}
    assert {"step", "attack", "eat", "pursue", "build", "say"} <= kinds


def test_a_thought_that_fails_is_kept_with_its_cause(tmp_path):
    """Trial 7: 75 thoughts failed and nothing said why."""
    import asyncio

    from nexel_agent.rl.mind import Mind
    m = Mind("Rook", "", "http://x", None, tmp_path / "m.json")
    asyncio.run(m.think({"self": {}}))
    log = (tmp_path / "mind_errors.log").read_text()
    assert m.stats["errors"] == 1 and "Error" in log and not m.busy
