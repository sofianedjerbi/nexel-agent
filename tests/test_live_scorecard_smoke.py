"""The live scorecard end to end (tools/rl_live_scorecard.py), two minutes, no model: a QA copy of the world, its --dev
server, the AI player and the scripted visitors, then the table. Slow and heavy: skipped unless NEXEL_SLOW=1, and only
when the Mac is free (one heavy job at a time). It leaves nothing: its run is removed and docs/kpis.* put back.

    NEXEL_SLOW=1 uv run pytest tests/test_live_scorecard_smoke.py -n 0 -s     (NEXEL_QA_WORLD=<a save> to copy another world)"""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORLD = Path(os.environ.get("NEXEL_QA_WORLD", ROOT.parent / "nexel-worlds" / "saves" / "caves"))   # (copied, never served; from a worktree: NEXEL_QA_WORLD)
KPIS = [ROOT / "docs" / "kpis.jsonl", ROOT / "docs" / "kpis.md"]

pytestmark = [pytest.mark.slow, pytest.mark.skipif(os.environ.get("NEXEL_SLOW") != "1", reason="slow: NEXEL_SLOW=1 to run it")]


def free_pair() -> int:
    """A port whose next one is free too (the server, and the player's brain feed on port + 1)."""
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        with socket.socket() as a, socket.socket() as b:
            if a.connect_ex(("127.0.0.1", p)) and b.connect_ex(("127.0.0.1", p + 1)) and p != 7777 and p + 1 != 7777:
                return p
    pytest.skip("no two free ports in a row")


@pytest.mark.skipif(not (WORLD / "events.jsonl").exists(), reason=f"no world to copy at {WORLD}")
def test_two_minutes_of_the_live_scorecard_with_no_model():
    kept = {p: p.read_bytes() if p.exists() else None for p in KPIS}
    runs = ROOT / "rl_runs" / "score"
    before = set(runs.iterdir()) if runs.exists() else set()
    out = None
    try:
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "rl_live_scorecard.py"), "--minutes", "2", "--model-url", "",
                            "--port", str(free_pair()), "--world", str(WORLD)], cwd=ROOT, capture_output=True, text=True, timeout=900)
        new = sorted(set(runs.iterdir()) - before) if runs.exists() else []
        assert r.returncode == 0 and len(new) == 1, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
        out = new[0]
        logs = out / "Rook"
        events = [json.loads(x) for x in (out / "saves" / "qa" / "events.jsonl").read_text().splitlines() if x.strip()]
        assert any(e.get("type") == "join" and e.get("name") == "Rook" for e in events)            # it came into the copy
        assert (logs / "rl.jsonl").stat().st_size > 0 and (logs / "brain.jsonl").stat().st_size > 0  # it played, and was watched
        assert "Traceback" not in (logs / "stderr.log").read_text()
        json.loads((logs / "report.json").read_text() or "{}")                                    # (its report: JSON)
        assert "asks" in json.loads((out / "visitor.json").read_text())                          # the visitors came and told
        card = (out / "scorecard.md").read_text()
        assert "the visitors saw" in card and "Rook" in card                                      # the table, with it in it
    finally:
        for p, b in kept.items():                                                                  # (the run's KPI row: not ours to keep)
            if b is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(b)
        if out is not None:
            shutil.rmtree(out, ignore_errors=True)
