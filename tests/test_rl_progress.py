"""The progress shaping (rl/progress.py): a long recipe chain pays as it goes, and over a goal it sums to nothing - a
goal that ends undone pays back all it paid, and what was dropped is paid back at once."""

from nexel_agent.rl.core import SimEnv
from nexel_agent.rl.goals import Goal
from nexel_agent.rl.progress import UNIT, left


def goals_with(item):
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    env.a.inventory.clear()
    g = env.goals
    g.goal = Goal("craft", item, None, env.sim.tick, 0, until=env.sim.tick + 10 ** 6)
    g.phi = 0.0
    return env, g


def test_steps_left_read_from_the_recipes():
    assert left("axe", {}, lambda s: False) > left("axe", {"wood": 6}, lambda s: False) > left("axe", {"wood": 2, "stone": 2, "fiber": 1}, lambda s: True) == 1


def test_progress_pays_as_it_goes_and_a_loss_is_paid_back():
    env, g = goals_with("axe")
    assert g.step([], {}, None) == 0                          # (set: nothing made yet)
    env.a.inventory["wood"] = 2
    gain = g.step([], {}, None)
    assert gain > 0
    env.a.inventory["wood"] = 0
    assert abs(g.step([], {}, None) + gain) < 1e-9            # (dropped: paid back)


def test_a_goal_that_ends_undone_sums_to_nothing():
    env, g = goals_with("axe")
    total = g.step([], {}, None)
    for k in ("wood", "stone"):
        env.a.inventory[k] = 2
        total += g.step([], {}, None)
    assert total >= 2 * UNIT
    g.goal.until = env.sim.tick                               # (time is up: another goal)
    total += g.step([], {}, None)
    assert abs(total) < 1e-9
