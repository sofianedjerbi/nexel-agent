"""The ship gate's live leg (tools/live_gate.py): a candidate passes only if it strikes first and kills no more than
the shipped body, on a QA copy with the scripted visitor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import live_gate  # noqa: E402


def test_a_body_that_kills_the_visitor_does_not_pass():
    shipped = {"fs_hour": 0.0, "killed": 0, "on_visitor": 0}
    assert live_gate.passes(shipped, {"fs_hour": 0.3, "killed": 0, "on_visitor": 0})
    assert not live_gate.passes(shipped, {"fs_hour": 1.94, "killed": 5, "on_visitor": 1})      # (r64, trial 8)
    assert not live_gate.passes(shipped, {"fs_hour": None, "killed": None, "on_visitor": None})  # (no scorecard: no pass)
