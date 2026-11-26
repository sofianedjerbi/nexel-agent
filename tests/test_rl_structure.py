"""Builds in three dimensions (#66): a design of storeys, a rectangle or a round, drawn (rl/structure.py), judged
storey by storey (rl/judge3d.py) and laid by the build verb alone in the game's own sim (rl/plan.py, rl/build.py)."""

import pytest

from nexel_agent.rl import architect, judge3d
from nexel_agent.rl.structure import UPSTAIRS_FURNITURE, Structure, footprint

HOME = {"bed": [0, 0], "campfire": [2, 0], "chest": [0, 2], "chair": [2, 2]}
DESIGNS = {
    "house": {"width": 3, "depth": 3, "door": "south", "furniture": HOME},
    "two storeys": {"width": 3, "depth": 3, "door": "south", "storeys": 2,
                    "furniture": {"campfire": [2, 0], "chest": [0, 2], "bed": [0, 0], "table": [1, 0, 1], "chair": [0, 0, 1]}},
    "round base": {"shape": "round", "radius": 4, "door": "south", "roof": False, "furniture": {"campfire": [3, 3]}},
    "round home": {"shape": "round", "radius": 3, "door": "south", "furniture": HOME},
    "tower": {"width": 3, "depth": 3, "door": "south", "storeys": 3,
              "furniture": {"campfire": [2, 0], "chest": [0, 2], "bed": [0, 0], "table": [0, 0, 1], "chair": [1, 0, 2]}},
}


@pytest.mark.parametrize("name", DESIGNS)
def test_each_kind_of_build_designed_well_passes_every_check(name):
    j = judge3d.judged(DESIGNS[name])
    assert j["ok"] and j["failed"] == [], (name, j["failed"])


def test_a_round_is_walled_whole_and_a_wall_gone_opens_it():
    inside, ring = footprint({"shape": "round", "radius": 4})
    near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]   # noqa: E731
    assert all(q in inside or q in ring for p in inside for q in near(p))           # (no way out but through a wall)
    s = Structure(DESIGNS["round base"])
    s.storeys[0].pop(next(p for p, t in s.storeys[0].items() if t == "wall"))
    from nexel_agent.rl.core import SimEnv
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    lv = judge3d.stamp(env.sim, s, env.a.x - 24, env.a.y - 24)
    assert "closed" in judge3d.judge_at(env.sim, s, env.a.x - 24, env.a.y - 24, lv)["failed"]


def test_upstairs_takes_only_what_the_game_stands_on_a_floor_and_says_so():
    assert "bed" in UPSTAIRS_FURNITURE and "campfire" not in UPSTAIRS_FURNITURE
    j = judge3d.judged({**DESIGNS["two storeys"], "furniture": {"campfire": [0, 0, 1]}})
    assert not j["ok"] and "needs the land under it" in j["error"]
    for bad, why in (({"storeys": 4}, "storeys from 1 to"), ({"shape": "star"}, "shape is one of"),
                     ({"shape": "round", "radius": 9}, "radius from 2 to 6")):
        assert why in judge3d.judged({**DESIGNS["house"], **bad})["error"]


def test_the_storeys_go_up_as_the_game_holds_them():
    """A floor's planks over the walls first, then each beside one laid; a flight of stairs from the ground to each
    upper storey's door, longer for each storey up."""
    s = Structure(DESIGNS["tower"])
    cells = s.cells()
    planks = [(x, y) for x, y, z, p in cells if p == "plank" and z == 3]
    walls = {p for p, t in s.storeys[0].items() if t in ("wall", "door")}
    seen = set()
    for i, (x, y) in enumerate(planks):
        assert (x, y) in walls or any(q in seen for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))), (i, x, y)
        seen.add((x, y))
    assert [len(steps) for _, steps, _, _ in s.stairs] == [2, 5]
    assert max(z for *_, z, p in cells if p == "roof") == 9


@pytest.mark.parametrize("name, seed", [("two storeys", 21), ("round base", 7), ("tower", 9)])
def test_a_build_of_storeys_or_a_round_is_laid_whole_by_building_alone(name, seed):
    b = judge3d.built(DESIGNS[name], seed=seed)
    assert b["sited"] and b["failed"] == [], b


def test_the_architect_chooses_a_build_of_storeys_and_hears_what_failed():
    import json
    replies = iter([json.dumps({**DESIGNS["two storeys"], "furniture": {"bed": [0, 0, 1]}}),
                    json.dumps(DESIGNS["two storeys"])])
    seen = []

    def ask(msgs):
        seen.append(msgs[-1]["content"])
        return next(replies)
    d = architect.design("a two-storey house", ask, budget=200)
    assert d["failed"] == [] and d["spec"] == DESIGNS["two storeys"], d["failed"]
    assert "no light" in seen[1] or "no chest" in seen[1], seen[1]
