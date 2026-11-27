"""The scenario benchmark (tools/scenarios.py) runs the body on goals from an empty pack: the goal's skill gets a
door made in a few decisions, as the live body now does (rl/live.py: a gather or craft goal's skill acts)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import scenarios  # noqa: E402


def test_the_goals_skill_crafts_a_door_from_an_empty_pack():
    pol = scenarios.NumpyPolicy(str(scenarios.POLICY))
    d = scenarios.run_goal(pol, "craft", "door", 1, 1000, "skills", 60)
    assert d is not None and d <= 20, d
    assert scenarios.wilson(12, 18) == [0.437, 0.837]
