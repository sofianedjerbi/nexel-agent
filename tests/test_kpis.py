"""The KPIs (tools/kpis.py): a training run's [check] line read into a row, a live scorecard's run read into a row, and
the page made from rows."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kpis as K  # noqa: E402

# a [check] line as train.py writes it (rl_runs/r*.out, step 480723)
CHECK = ("[check] step=480723 min=30 sps=254 score=5.747 r6set=16.83 (random 1.081, legal random 1.563, noop 0.036) ach=71.19 "
         "ticks=9011 deaths=0/48 crafts_distinct=23.25 places=98.8 places_distinct=18.73 nights=1.0 chunks=8.3 fail%=20.8 "
         "fights=61W/0L/2F of 63 rival=131W/17F/72E/20K social=5.96/6 hit_neighbour=0 unprovoked=0 first_strikes=0 "
         "near_min=296.1 fs/near_min=0.0 attack_picks=[back 153 / slot 0] said/ep=12.7 goals=[gather:22/32 craft:28/37 "
         "give:18/18 ask:18/21 stay:15/30 help:34/89 avoid:4/20] hits=246 kl_mean=0.0004 kl_max=0.0030 entropy=1.688 "
         "tool_p=0.0482 (n=552) gear=[spear 0.12 axe 0 pickaxe 0 armed_blows 0.05 armor 0 fiber_at_bench 0.06 "
         "repeat_crafts 0.77] fills=0.44/0.12/0.02 stock=0.67/0.08 multi=0.98 rooms=0.12 build=0.19/0.0 train_score=6.465 "
         "train_ach=74.53")


def test_a_check_line_is_read_into_its_row():
    r = K.sim_row("r60", 480723, "lush", CHECK)
    assert r == {"kind": "sim", "run": "r60", "step": 480723, "set": "lush", "score": 5.747, "deaths": 0, "fs_near_min": 0.0,
                 "goals_done": round(139 / 247, 3), "tools": 0.12, "repeat_crafts": 0.77, "rooms": 0.12,
                 "build_grade": 0.19, "builds_5": 0.0, "fail_pct": 20.8}


def test_a_line_without_goals_gear_or_builds_leaves_them_empty():
    """(older runs wrote fewer fields: what is not there is None, not a guess)"""
    r = K.sim_row("r33", 1000, "live-like", "[live-like] score=2.5 deaths=3/24 fs/near_min=0.4 fail%=31.0")
    assert (r["score"], r["deaths"], r["fs_near_min"], r["fail_pct"]) == (2.5, 3, 0.4, 31.0)
    assert r["goals_done"] is r["tools"] is r["repeat_crafts"] is r["rooms"] is r["build_grade"] is r["builds_5"] is None


RUN = K.ROOT / "rl_runs" / "score" / "20261125-181950"      # (a live scorecard's run on this machine: not in git)


@pytest.mark.skipif(not RUN.exists(), reason="the scorecard run rl_runs/score/20261125-181950 is not on this machine")
def test_a_live_scorecard_run_is_read_into_its_row():
    r = K.live_row(RUN)
    assert r == {"kind": "live", "run": "20261125-181950", "who": "Rook", "model": "words", "minutes": 31, "lines_min": 0.81,
                 "reply_p90_s": 2.56, "fs_hour": 0.0, "repeat_crafts_pct": 85, "tools_made": 6, "fail_pct": 23.0,
                 "asks_met": "6/6", "builds_judged": 0, "best_grade": None, "best_sense": None, "best_looks": None,
                 "players_builds_judged": 3, "players_best_grade": 4, "players_best_sense": 5, "players_best_looks": 6}


LIVE = {"kind": "live", "run": "20261125-181950", "who": "Rook", "model": "words", "minutes": 31, "lines_min": 0.81,
        "reply_p90_s": 2.56, "fs_hour": 0.0, "repeat_crafts_pct": 85, "tools_made": 6, "fail_pct": 23.0, "asks_met": "6/6",
        "builds_judged": 0, "best_grade": None, "best_sense": None, "best_looks": None,
        "players_builds_judged": 3, "players_best_grade": 4, "players_best_sense": 5, "players_best_looks": 6}


def test_the_page_has_a_table_per_set_and_live_rows_with_the_players_builds_beside():
    rows = [K.sim_row("r60", 480723, "lush", CHECK), K.sim_row("r60", 480723, "live-like", CHECK), LIVE]
    page = K.show(rows)
    lines = page.splitlines()
    head = next(x for x in lines if x.startswith("| run | step |"))
    assert "deaths ↓" in head and "score ↑" in head and head.count("|") == len(K.SIM) + 3      # (an arrow: which way is better)
    sims = [x for x in lines if x.startswith("| r60 | 480723 |")]
    assert len(sims) == 2 and "| 5.747 |" in sims[0] and "| 0.563 |" in sims[0]                  # (one row in each set's table)
    assert "| 20261125-181950 | words | 31 | 0.81 | 2.56 |" in page
    builds = next(x for x in lines if x.startswith("| 20261125-181950 | 0 |"))
    assert builds.endswith("| - | - | - | 3 | 4 | 5 | 6 |")                                       # (none of its own judged; the players' beside)
    assert "## Body in the sim: lush held-out worlds" in page and "## Body in the sim: live-like held-out worlds" in page
