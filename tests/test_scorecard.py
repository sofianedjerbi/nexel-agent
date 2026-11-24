"""The live scorecard (tools/scorecard.py): rows read from the game's events, the same for a person and an AI player."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import scorecard as S  # noqa: E402

M = 60 * S.TPS


def test_rows_from_the_games_events():
    ev = [{"tick": 0, "type": "join", "name": "Rook"}, {"tick": 0, "type": "join", "name": "mira"},
          {"tick": 10, "type": "say", "from": "mira", "text": "hey rook", "x": 0, "y": 0},
          {"tick": 30, "type": "say", "from": "Rook", "text": "Hey there friend", "x": 1, "y": 0, "bot": True},
          {"tick": M, "type": "craft", "name": "Rook", "made": {"plank": 2}},
          {"tick": M + 1, "type": "craft", "name": "Rook", "made": {"plank": 2}},
          {"tick": 2 * M, "type": "craft", "name": "Rook", "made": {"axe": 1}},
          {"tick": 2 * M + 5, "type": "attack", "attacker": "Rook", "target": "mira", "damage": 1},
          {"tick": 2 * M + 6, "type": "attack", "attacker": "mira", "target": "Rook", "damage": 1},     # (hits back: no first strike)
          {"tick": 2 * M + 7, "type": "attack", "attacker": "Rook", "target": "chicken#4", "damage": 1},
          {"tick": 3 * M, "type": "death", "name": "mira", "by": "Rook"},
          {"tick": 3 * M + 1, "type": "give", "from": "Rook", "to": "mira", "items": {"wood": 1}}]
    P = S.players(ev)
    r = S.row(P["Rook"])
    assert P["Rook"]["bot"] and not P["mira"]["bot"]
    assert r["active min"] == 4 and r["reply s (p50)"] == 2.0 and r["lowercase lines %"] == 0
    assert (r["distinct crafts"], r["repeat crafts %"], r["tools or weapons made"]) == (2, 33, 1)
    assert (r["first strikes on people / hour"], r["people killed"], r["gifts given"]) == (15.0, 1, 1)
    assert S.row(P["mira"])["first strikes on people / hour"] == 0
