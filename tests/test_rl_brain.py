"""The brain feed (rl/brain.py, docs/brain_feed.md): fake observations in, feed lines out, and the SSE server."""

import json
import socket
import time
from types import SimpleNamespace

from nexel_agent.rl.brain import Brain, chat_or_event, crop, frame, sse, top
from nexel_agent.rl.encode import action_name
from nexel_agent.rl.goals import Goal


def grid(n=21):
    rows = [list("." * n) for _ in range(n)]
    rows[n // 2][n // 2] = "@"
    rows[n // 2 - 1][n // 2 + 2] = "P"
    rows[0][0] = "T"                      # outside the 13 x 13
    return ["".join(r) for r in rows]


def obs(**kw):
    o = {"tick": 100, "time": "day 3, 17:02",
         "self": {"x": 7, "y": -35, "z": 2, "hp": 17, "hunger": 16, "deaths": 0, "inventory": {"wood": 3, "dirt": 2}},
         "grid": grid(),
         "entities": [{"id": 5, "kind": "agent", "name": "Mira", "dx": 2, "dy": -1, "hp": 14, "holding": "sword"},
                      {"id": 9, "kind": "sporeling", "dx": -3, "dy": 0, "hp": 2}],
         "events": []}
    o.update(kw)
    return o


def test_crop_is_the_middle():
    t = crop(grid())
    assert len(t) == 13 and all(len(r) == 13 for r in t)
    assert t[6][6] == "@" and t[5][8] == "P" and "T" not in "".join(t)


def test_frame_fields():
    f = frame(obs(), "W", "dig_W", [1, 0, 1, 1], Goal("gather", "wood", "Mira"), 12.4, {"Mira"}, set())
    assert f["k"] == "frame" and f["pos"] == [7, -35, 2] and (f["hp"], f["food"], f["facing"]) == (17, 16, "W")
    assert f["time"] == "day 3, 17:02" and len(f["tiles"]) == 13
    assert f["beings"][0] == {"name": "Mira", "kind": "agent", "dx": 2, "dy": -1, "hp": 14, "side": "ally"}
    assert f["beings"][1]["side"] == "foe" and "name" not in f["beings"][1]
    assert f["goal"] == {"kind": "gather", "item": "wood", "have": 3, "with": "Mira", "since_s": 12}
    assert (f["act"], f["legal"], f["of"]) == ("dig_W", 3, 4)
    json.dumps(f)                                                   # (plain JSON)


def test_frame_leaves_out_what_it_does_not_know():
    o = obs()
    del o["grid"], o["time"]
    f = frame(o, goal=Goal())
    assert "tiles" not in f and "time" not in f and "goal" not in f and "top" not in f and "legal" not in f


def test_top_softmax_after_the_mask():
    t = top([0.0, 5.0, 1.0, 9.0, 2.0, 3.0, 4.0, -1.0], mask=[1, 1, 1, 0, 1, 1, 1, 1], names=lambda i: f"a{i}")
    assert [n for n, _ in t] == ["a1", "a6", "a5", "a4", "a2", "a0"]          # a3 is masked
    assert 0.5 < t[0][1] < 1 and abs(sum(p for _, p in top([1.0, 2.0, 3.0])) - 1) < 0.01


def test_chat_and_events():
    assert chat_or_event({"type": "say", "from": "Mira", "text": "hi"}, "Rook") == {"k": "chat", "from": "Mira", "text": "hi"}
    assert chat_or_event({"type": "say", "from": "Rook", "text": "yo"}, "Rook")["said"] is True
    e = chat_or_event({"type": "attack", "attacker": "sporeling", "target": "Rook", "damage": 1}, "Rook")
    assert e == {"k": "event", "what": "hurt", "text": "sporeling hit Rook (1)", "n": 1}
    assert chat_or_event({"type": "tile"}, "Rook") is None
    assert chat_or_event({"type": "craft", "name": "Mira", "made": {"plank": 2}}, "Rook") is None
    assert chat_or_event({"type": "craft", "name": "Rook", "made": {"plank": 2}}, "Rook")["text"] == "Rook made 2 plank"


class FakeMind:
    def __init__(self):
        self.relations, self.goal, self.goal_at = {"Mira": {"ally": True, "notes": "shares seeds"}}, Goal("gather", "wood"), time.monotonic()

    def facts(self, o):
        return "You: health 17/20."

    def apply(self, d):
        self.goal = Goal(d["goal"], d["item"])
        self.relations["Vex"] = {"ally": False, "notes": d["relation_update"]["Vex"]["note"]}


class FakePolicy:
    def logits(self, enc):
        return [0.0, 3.0, 1.0, 2.0]


def body(mind=None):
    return SimpleNamespace(mind=mind, facing="S", hurt=0.0, foes={}, pol=FakePolicy())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_lines_in_the_file(tmp_path):
    p = tmp_path / "brain.jsonl"
    b = Brain(None, p, "Rook", "a quiet player")
    m = FakeMind()
    bd = body(m)
    b.see(obs(events=[{"type": "say", "from": "Mira", "text": "got any stone?"}]), bd, enc={}, mask=[1, 1, 0, 1], i=None)
    b.see(obs(), bd)                                             # (within the second: no second frame)
    m.facts(obs())
    m.apply({"goal": "gather", "item": "wood", "relation_update": {"Vex": {"note": "never paid back the iron"}}})
    b.close()
    ls = lines(p)
    kinds = [x["k"] for x in ls]
    assert kinds[0] == "hello" and ls[0]["name"] == "Rook" and ls[0]["v"] == 1
    assert kinds.count("frame") == 1 and "chat" in kinds and kinds.count("thought") == 1 and kinds.count("memory") == 2
    f = next(x for x in ls if x["k"] == "frame")
    assert f["top"][0][1] == 0.705 and f["top"][0][0] == action_name(1) and len(f["top"]) == 3 and f["legal"] == 3 and f["goal"]["kind"] == "gather"
    th = next(x for x in ls if x["k"] == "thought")
    assert th["facts"] == "You: health 17/20." and th["reply"]["item"] == "wood" and th["took_s"] >= 0
    assert ls[-1]["k"] == "memory" and ls[-1]["people"]["Vex"]["notes"] == "never paid back the iron"
    assert {"k": "event", "what": "goal", "text": "goal: gather wood"}.items() <= next(x for x in ls if x["k"] == "event").items()
    assert all("t" in x for x in ls)


def test_see_never_raises(tmp_path):
    b = Brain(None, tmp_path / "b.jsonl", "Rook")
    b.see({"self": None}, object())                              # (garbage in: nothing out, no error)
    b.close()


def test_sse_format():
    assert sse('{"k":"frame"}') == b'data: {"k":"frame"}\n\n'


def read_events(port, n, timeout=3.0):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.sendall(b"GET /feed HTTP/1.1\r\nHost: x\r\n\r\n")
    buf, end = b"", time.monotonic() + timeout
    while buf.count(b"\n\ndata: ") + 1 < n + 1 and time.monotonic() < end:
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        buf += chunk
    s.close()
    head, _, body = buf.partition(b"\r\n\r\n")
    return head.decode(), [json.loads(x[6:]) for x in body.decode().split("\n\n") if x.startswith("data: ")]


def test_serves_sse(tmp_path):
    b = Brain(0, None, "Rook")
    try:
        b.emit({"k": "chat", "from": "Mira", "text": "hi"})
        b.flush()
        head, evs = read_events(b.port, 2)
        assert "200" in head.splitlines()[0] and "text/event-stream" in head and "Access-Control-Allow-Origin: *" in head
        assert [e["k"] for e in evs[:2]] == ["hello", "chat"]
    finally:
        b.close()


def test_late_page_gets_hello_and_memory_first(tmp_path, monkeypatch):
    import nexel_agent.rl.brain as brain
    monkeypatch.setattr(brain, "BACKLOG", 5)
    b = Brain(0, None, "Rook")
    try:
        b.attach(FakeMind())
        for i in range(20):
            b.emit({"k": "event", "what": "x", "text": str(i)})
        b.flush()
        _, evs = read_events(b.port, 7)
        assert [e["k"] for e in evs[:2]] == ["hello", "memory"] and evs[-1]["text"] == "19" and len(evs) == 7
    finally:
        b.close()


def test_the_goals_tally_goes_out_when_it_changes(tmp_path):
    """How each goal the mind set went (mind.tried), checked with the frame once a second: a line when it changed only."""
    p = tmp_path / "brain.jsonl"
    b = Brain(None, p, "Rook")
    m = FakeMind()
    m.tried = {"gather wood": [1, 1]}
    bd = body(m)
    b.see(obs(), bd)
    b.t_frame = 0                                               # (a second on: the next frame)
    b.see(obs(), bd)                                            # (the same tally: no new line)
    m.tried["make a bed"] = [1, 0]
    b.t_frame = 0
    b.see(obs(), bd)
    b.close()
    tallies = [x for x in lines(p) if x["k"] == "tally"]
    assert [x["goals"] for x in tallies] == [{"gather wood": [1, 1]}, {"gather wood": [1, 1], "make a bed": [1, 0]}]


def test_a_late_page_gets_the_last_tally_first(monkeypatch):
    """(kept for a page that connects after it went out of the backlog, as hello and memory are)"""
    import nexel_agent.rl.brain as brain
    monkeypatch.setattr(brain, "BACKLOG", 5)
    b = Brain(0, None, "Rook")
    try:
        m = FakeMind()
        m.tried = {"gather wood": [2, 1]}
        b.see(obs(), body(m))
        for i in range(20):
            b.emit({"k": "event", "what": "x", "text": str(i)})
        b.flush()
        _, evs = read_events(b.port, 8)
        assert [e["k"] for e in evs[:3]] == ["hello", "memory", "tally"] and evs[2]["goals"] == {"gather wood": [2, 1]}
    finally:
        b.close()


def test_no_tally_before_a_goal_ended(tmp_path):
    p = tmp_path / "brain.jsonl"
    b = Brain(None, p, "Rook")
    m = FakeMind()
    m.tried = {}
    b.see(obs(), body(m))
    b.close()
    assert "tally" not in [x["k"] for x in lines(p)]


def test_the_home_the_mind_is_told_goes_out_when_it_changes(tmp_path):
    """mind.home_read (rl/home.py reading), as it was given to the mind: none while no home is chosen, then each new
    reading once; kept for a late page."""
    p = tmp_path / "brain.jsonl"
    b = Brain(None, p, "Rook")
    m = FakeMind()
    m.home, m.home_read = None, None
    bd = body(m)
    b.see(obs(), bd)
    m.home, m.home_read = (3, 4), {"at": [3, 4], "where": "2 tiles N", "far": False, "closed": False, "told": "not closed",
                                   "verdict": {"grade": 0, "of": 6, "checks": {}, "failed": []}, "lacks": {"wood": 9}}
    for _ in range(2):
        b.t_frame = 0
        b.see(obs(), bd)
    b.close()
    homes = [x for x in lines(p) if x["k"] == "home"]
    assert [h.get("none") for h in homes] == [True, None] and homes[1]["where"] == "2 tiles N" and homes[1]["lacks"] == {"wood": 9}

