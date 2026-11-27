"""Houses of wings (rl/wings.py): rectangles joined into one inside - a T, a cross, an L - each wing under its own
gable, judged and laid as any build of storeys; the words "L-shaped", "a cross", "wings" ask for one."""

import pytest

from nexel_agent.rl import judge3d
from nexel_agent.rl.asked import asked, unlike
from nexel_agent.rl.structure import Structure

WINGS = {"T": {"shape": "wings", "wings": [[0, 0, 7, 3], [2, 3, 3, 3]], "door": "south",
               "roof": {"style": "gable", "material": "dirt", "ridge": "plank"},
               "furniture": {"bed": [0, 0], "campfire": [2, 0], "chest": [6, 0], "table": [3, 4]}},
         "cross": {"shape": "wings", "wings": [[0, 2, 7, 3], [2, 0, 3, 7]], "door": "south",
                   "roof": {"style": "hip", "material": "plank", "eaves": "stone_block"},
                   "furniture": {"bed": [2, 2], "campfire": [4, 2], "chest": [3, 0], "table": [0, 3]}}}


@pytest.mark.parametrize("name", WINGS)
def test_a_house_of_wings_passes_every_check_and_is_laid_whole_by_building_alone(name):
    assert judge3d.judged(WINGS[name])["failed"] == []
    b = judge3d.built(WINGS[name], seed=7)
    assert b["sited"] and b["failed"] == [], b


def test_each_wing_has_its_own_ridge_and_wings_that_do_not_touch_are_refused():
    s = Structure(WINGS["T"])
    tops = {(x, y) for x, y, z, _ in s.cells() if z == max(c[2] for c in s.cells())}
    assert {y for _, y in tops} != {x for x, _ in tops} and len({x for x, _ in tops}) > 1 and len({y for _, y in tops}) > 1
    with pytest.raises(ValueError, match="touch"):
        Structure({**WINGS["T"], "wings": [[0, 0, 3, 3], [5, 5, 3, 3]]})
    with pytest.raises(ValueError, match="2 to 4"):
        Structure({**WINGS["T"], "wings": [[0, 0, 3, 3]]})


def test_the_words_for_an_l_or_a_cross_ask_for_wings():
    assert asked("an L-shaped house with a gabled roof") == {"roof": "gable", "shape": "wings"}
    assert asked("a cross shaped hall")["shape"] == "wings" and "shape" not in asked("will you help me")
    assert "wings" in unlike({"width": 5, "depth": 3, "roof": {"style": "gable"}}, asked("an L-shaped house"))
    assert unlike(WINGS["T"], asked("an L-shaped house")) == ""
