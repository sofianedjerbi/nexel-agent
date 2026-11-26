"""The builds a save holds, judged (tools/judge_builds.py, #27): each closed room is the player's or the real
players', and the live scorecard and its KPIs read the best grades of each."""

import json
import sys
from pathlib import Path

from nexel_worlds.sim.core import Simulation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import judge_builds as J  # noqa: E402


def tile(x, y, block, by, cause="place"):
    return json.dumps({"type": "tile", "x": x, "y": y, "block": block, "by": by, "cause": cause})


def test_what_a_player_laid_and_still_stands(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text("\n".join([tile(1, 1, "gate", "Rook"), tile(1, 1, "none", "Rook", "dig"),       # laid, then dug
                             tile(2, 2, "wood_wall", "Rook"), tile(3, 3, "wood_wall", "Mira"),     # its own; another's
                             tile(4, 4, "chair", "Rook")]) + "\n")                                  # (no wall)
    assert J.placed_by(ev, "Rook") == {(2, 2)}


def test_a_save_s_rooms_are_judged_and_told_apart(tmp_path):
    """Two rooms in a save: one whose wall the player laid a piece of, one nobody named laid (the world's people)."""
    db, ev = tmp_path / "world.sqlite", tmp_path / "events.jsonl"
    sim = Simulation(db, seed=12, event_log=tmp_path / "sim.jsonl")
    rooms = [(0, 0), (20, 0)]
    for ox, oy in rooms:                                   # 3 by 3 inside, a door in the north wall
        for i in range(-1, 4):
            for j in range(-1, 4):
                sim.world.set(ox + i, oy + j, ground="grass", block="none")
                if i in (-1, 3) or j in (-1, 3):
                    sim.world.set(ox + i, oy + j, block="door" if (i, j) == (1, -1) else "wood_wall")
    sim.flush()
    sim.close()
    ev.write_text(tile(-1, 1, "wood_wall", "Rook") + "\n")
    rows = J.judged(db, ev, "Rook")
    assert sorted(r["by"] for r in rows) == ["Rook", "players"]
    mine = J.summary(rows, "Rook")
    assert mine["builds"] == 1 and mine["best_grade"] >= 3 and mine["best_sense"] is not None and mine["best_looks"] is not None
    assert J.summary(rows, "Mira") == {"builds": 0, "best_grade": None, "best_sense": None, "best_looks": None}
    card = J.table(rows, "Rook")
    assert "| Rook | 1 |" in card and "| real players | 1 |" in card


def test_the_kpi_page_shows_its_builds_beside_the_players():
    import kpis
    row = {"kind": "live", "run": "r1", "who": "Rook", "model": "words", "minutes": 30, "builds_judged": 1, "best_grade": 4,
           "best_sense": 5, "best_looks": None, "players_builds_judged": 3, "players_best_grade": 4,
           "players_best_sense": 5, "players_best_looks": 6}
    page = kpis.show([row])
    assert "## Live builds" in page and "| r1 | 1 | 4 | 5 | - | 3 | 4 | 5 | 6 |" in page
