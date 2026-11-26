"""The ship gate (tools/ship_gate.py): its numbers and its verdict, from passes made up here (no body is played: that
needs torch, the trainer's)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ship_gate as G  # noqa: E402


def test_the_interval_is_students_t_on_the_passes():
    m, h = G.ci([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and h == pytest.approx(3.18 * (5 / 3) ** .5 / 2)      # (n 4: t 3.18, stdev 1.29, over sqrt 4)
    assert G.ci([1.0, 2.0])[1] == pytest.approx(12.71 * 2 ** -.5 / 2 ** .5)
    assert G.ci([5.0]) == (5.0, float("inf"))                             # (one pass: no interval, nothing is sure)
    assert G.ci([1.0] * 7)[1] == 0                                        # (an n with no t of its own: 2.0, no spread)


def test_better_same_and_worse_by_the_interval_of_the_difference():
    base = [1.0, 1.1, 0.9, 1.0]
    assert G.verdict("score", base, [2.0, 2.1, 1.9, 2.0]) == "better"
    assert G.verdict("score", base, [0.0, 0.1, -0.1, 0.0]) == "worse"
    assert G.verdict("score", base, [1.05, 0.95, 1.1, 0.9]) == "same"     # (the intervals overlap zero)
    assert G.verdict("score", [1.0], [9.0]) == "same"                     # (one pass each: never sure)


@pytest.mark.parametrize("k", sorted(G.LOWER_IS_BETTER))
def test_for_a_row_where_lower_is_better_fewer_is_better(k):
    base = [10.0, 11.0, 9.0, 10.0]
    assert G.verdict(k, base, [1.0, 1.1, 0.9, 1.0]) == "better"
    assert G.verdict(k, base, [20.0, 21.0, 19.0, 20.0]) == "worse"


def passes(**rows):
    """{set: [pass rows]}: each row k the values given, one a pass (one set, 'lush')."""
    n = len(next(iter(rows.values())))
    return {"lush": [{k: v[i] for k, v in rows.items()} for i in range(n)]}


SAME = [1.0, 1.1, 0.9, 1.0]
UP = [2.0, 2.1, 1.9, 2.0]
DOWN = [0.0, 0.1, -0.1, 0.0]


def test_it_ships_when_the_target_is_better_and_no_guard_is_worse():
    B = passes(score=SAME, deaths=SAME, fs_per_near_min=SAME, goals_done=SAME)
    C = passes(score=SAME, deaths=SAME, fs_per_near_min=SAME, goals_done=UP)
    lines, ok = G.judge(B, C, ["goals_done"])
    assert ok and len(lines) == 4 and lines[3].startswith("| lush goals_done | 1.000 ±") and lines[3].endswith("| better |")


def test_a_target_only_the_same_does_not_ship():
    B = passes(score=SAME, deaths=SAME, fs_per_near_min=SAME, goals_done=SAME)
    assert not G.judge(B, passes(score=SAME, deaths=SAME, fs_per_near_min=SAME, goals_done=SAME), ["goals_done"])[1]


@pytest.mark.parametrize("guard", G.GUARDS)
def test_a_guard_worse_keeps_the_shipped_body_even_with_the_target_better(guard):
    B = passes(score=SAME, deaths=SAME, fs_per_near_min=SAME, goals_done=SAME)
    worse = UP if guard in G.LOWER_IS_BETTER else DOWN
    C = passes(**{"score": SAME, "deaths": SAME, "fs_per_near_min": SAME, "goals_done": UP, guard: worse})
    lines, ok = G.judge(B, C, ["goals_done"])
    assert not ok and any(x.startswith(f"| lush {guard} |") and x.endswith("| worse |") for x in lines)


def test_by_default_the_target_is_the_score_and_a_row_neither_target_nor_guard_does_not_count():
    B = passes(score=SAME, deaths=SAME, fs_per_near_min=SAME, rooms=SAME)
    C = passes(score=UP, deaths=SAME, fs_per_near_min=SAME, rooms=DOWN)
    assert G.judge(B, C, ["score"])[1]
    assert not G.judge(B, passes(score=SAME, deaths=SAME, fs_per_near_min=SAME, rooms=UP), ["score"])[1]


def test_every_set_is_judged():
    B = {**passes(score=SAME, deaths=SAME, fs_per_near_min=SAME), "live-like": passes(score=SAME, deaths=SAME, fs_per_near_min=SAME)["lush"]}
    C = {**passes(score=UP, deaths=SAME, fs_per_near_min=SAME), "live-like": passes(score=UP, deaths=UP, fs_per_near_min=SAME)["lush"]}
    lines, ok = G.judge(B, C, ["score"])
    assert not ok and len(lines) == 6 and lines[4] == next(x for x in lines if x.startswith("| live-like deaths |"))


METRICS = {"score": 5.7, "deaths": 1, "fs_per_near_min": .02, "crafts_distinct": 23.2, "repeat_crafts": .77, "fail_pct": 20.8,
           "top": {"craft_spear": 3, "craft_axe": 2, "craft_pickaxe": 1, "craft_sword": 0, "craft_plank": 9, "room_built": 2},
           "goals": "gather:22/32 craft:28/37 avoid:0/0", "build_grade": .19}


def test_the_rows_of_a_pass_from_evaluates_metrics():
    r = G.rows(METRICS)
    assert {k: r[k] for k in ("score", "deaths", "fs_per_near_min", "crafts_distinct", "repeat_crafts", "fail_pct")} == \
        {"score": 5.7, "deaths": 1, "fs_per_near_min": .02, "crafts_distinct": 23.2, "repeat_crafts": .77, "fail_pct": 20.8}
    assert r["tools_made"] == 6 and r["rooms"] == 2                         # (spear, axe, pickaxe, sword, hoe: not planks)
    assert r["build_grade"] == .19 and r["builds_5"] == 0                   # (a metric not there: 0)
    assert r["goals_done"] == round(50 / 69, 3)                             # (all the goals together)
    assert (r["goal_gather"], r["goal_craft"], r["goal_avoid"]) == (round(22 / 32, 3), round(28 / 37, 3), 0)   # (each; none set: 0, not a division by 0)


class Masks:
    """A venv's masks: one per env, as long as the actions of today."""

    def __init__(self, n_env, n_act):
        self.masks = [np.arange(n_act) % 2 == 0 for _ in range(n_env)]


def test_an_older_body_is_shown_its_own_part_of_the_vector_and_of_the_masks():
    o = {"vec": np.arange(2 * 40, dtype=float).reshape(2, 40), "grid": np.ones((2, 3, 3))}
    v = Masks(2, 50)
    obs, masks = G.trim(o, v.masks, 32, 45)
    assert obs["vec"].shape == (2, 32) and (obs["vec"] == o["vec"][:, :32]).all()   # (the first 32: what it was trained on)
    assert obs["grid"] is o["grid"] and o["vec"].shape == (2, 40)                   # (the rest as it is; the env's own untouched)
    assert masks.shape == (2, 45) and (masks == np.stack(v.masks)[:, :45]).all()
    same, m2 = G.trim(o, v.masks, 40, 50)
    assert same["vec"].shape == (2, 40) and m2.shape == (2, 50)                     # (a body of today: all of it)
