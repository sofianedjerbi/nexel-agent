"""A build's renders (tools/render_build.py) on the review page (tools/build_review.py), and the vision model's rubric
(tools/build_vlm.py) without a model: what it asks, what it reads of an answer, what it gives back (#32)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_review as B  # noqa: E402
import build_vlm as V  # noqa: E402
import render_build as R  # noqa: E402

PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000")   # (the head of a 1x1 PNG: only sent, never read)
ROOM = {"at": [5, 9], "by": "players", "v1": 1, "v2": 1, "v3": 1, "of": [1, 1, 1], "v1_failed": [], "v2_failed": [], "v3_failed": [],
        "facts": {}, "looks": {},
        "map": {"x0": 2, "y0": 6, "blocks": [".....", ".....", "....."], "grounds": ["....."] * 3, "roofs": ["00000"] * 3,
                "inside": ["00000", "01110", "00000"]}}


def test_a_room_s_renders_are_named_by_run_and_door_and_shown_beside_its_grid(tmp_path):
    assert R.render_name("r1", ROOM, "top") == "r1_5_9_top.png"
    assert R.centre(ROOM["map"]) == (4, 7)                                 # (its inside tiles: x 3..5, y 7)
    page = tmp_path / "index.html"
    assert "<img" not in B.room(ROOM, "r1", page)                          # none made: none shown
    (tmp_path / "renders").mkdir()
    for v in R.VIEWS:
        (tmp_path / "renders" / R.render_name("r1", ROOM, v)).write_bytes(PNG)
    shown = B.room(ROOM, "r1", page)
    assert all(f'src="renders/r1_5_9_{v}.png"' in shown for v in R.VIEWS)


def test_the_rubric_asks_with_the_pictures_and_reads_scores_and_reasons(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(PNG)
    req = V.request([p, p])
    parts = req["messages"][0]["content"]
    assert req["temperature"] == 0 and parts[0]["text"] == V.PROMPT and len(parts) == 3
    assert all(q["image_url"]["url"].startswith("data:image/png;base64,") for q in parts[1:])
    answer = '```json\n{"neat": {"score": 4, "reason": "straight"}, "cosy": {"score": 2, "reason": "empty"}, "purposeful": {"score": 3, "reason": "ok"}}\n```'
    assert V.parse(answer) == {"neat": {"score": 4, "reason": "straight"}, "cosy": {"score": 2, "reason": "empty"},
                               "purposeful": {"score": 3, "reason": "ok"}}
    trailing = '{"neat": {"score": 3, "reason": "a",}, "cosy": {"score": 2, "reason": "b"}, "purposeful": {"score": 2, "reason": "c"},}'
    assert V.parse(trailing)["neat"] == {"score": 3, "reason": "a"}             # (a 3B model's answer, 2 rooms in 5)
    with pytest.raises(ValueError):
        V.parse('{"neat": {"score": 7}, "cosy": {"score": 2}, "purposeful": {"score": 3}}')   # (out of 1-5)
    with pytest.raises(ValueError):
        V.parse("I like it")


def test_the_rubric_gives_each_criterion_and_their_mean_from_the_model_s_answer(tmp_path, monkeypatch):
    p = tmp_path / "a.png"
    p.write_bytes(PNG)
    sent = {}

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            text = json.dumps({c: {"score": s, "reason": c} for c, s in zip(V.CRITERIA, (5, 4, 3), strict=True)})
            return {"choices": [{"message": {"content": text}}]}

    monkeypatch.setattr(V.httpx, "post", lambda url, json, timeout: sent.update(url=url, body=json) or Reply())
    got = V.rubric([p], "http://vlm:8080/v1/", model="m")
    assert sent["url"] == "http://vlm:8080/v1/chat/completions" and sent["body"]["model"] == "m"
    assert got["neat"]["score"] == 5 and got["mean"] == 4 and got["rubric"] == V.RUBRIC_VERSION and got["model"] == "m"
