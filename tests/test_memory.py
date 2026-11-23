"""The player's own memory (memory.py) and what it sees of danger (senses.py), with the fake game and model:
writes, retrieval by person and place, a restart, the diary that consolidates, a chat that recalls a hit."""

import asyncio
import json

import httpx

from megastructure.memory import MAX_EPISODES, Memory
from megastructure.senses import Body, danger_lines
from test_thinker_realtime import FakeGame, fake_obs, model_transport, thinker


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_notes_and_episodes_survive_a_restart(tmp_path):
    clock = Clock()
    m = Memory(tmp_path / "memory.json", clock)
    n = m.remember("my chest is by the pond", (5, 6), "day 1, 10:00")
    m.heard("Ana", "can you build me a fence?", (2, 2), "day 1, 10:01")
    m.hit_by("Bo", 3, "sword", (0, 0), "day 1, 10:02", hp=14)
    again = Memory(tmp_path / "memory.json", clock)
    assert f"[{n}] my chest is by the pond (5,6)" in again.notes_text()
    assert 'they said: "can you build me a fence?"' in again.person_text("Ana")
    assert "they hit you 1x" in again.person_text("Bo")
    assert again.forget(n) and again.notes_text() == "" and not again.forget(n)


def test_retrieval_by_person_and_place_and_goal():
    clock = Clock()
    m = Memory(None, clock)
    m.add("place", "you placed a workbench", None, (100, 100))
    m.heard("Ana", "the cave east has ore", (0, 0))
    for i in range(10):
        m.add("misc", f"walked about {i}", None, (500 + i, 500))
    clock.t += 3600
    assert any("Ana" in x for x in m.relevant("", (300, 300), ["Ana"], k=2))           # the person in view
    assert any("workbench" in x for x in m.relevant("", (101, 99), [], k=2))           # the place you are at
    assert any("ore" in x for x in m.relevant("mine some ore", (300, 300), [], k=2))   # the goal's words
    assert m.relevant("", None, [], k=3)[0].startswith("60 min ago")


def test_a_hit_every_second_is_one_fight_counted():
    m = Memory(None, Clock())
    for _ in range(4):
        m.add("fight", "a tusker hit you with a bite (-3)", "a tusker", (0, 0))
    assert len(m.episodes) == 1 and m.relevant()[0].endswith("(x4)")


def test_the_diary_consolidates_the_oldest_episodes_when_full(tmp_path):
    clock = Clock()
    m = Memory(tmp_path / "memory.json", clock)
    for i in range(MAX_EPISODES + 5):
        m.add("misc", f"event {i}", None, (i, 0))
    assert m.summary_due()
    msgs, n = m.summary_request("Vex")
    assert n == 30 and "event 0" in msgs[1]["content"] and "event 29" in msgs[1]["content"] and "event 30" not in msgs[1]["content"]
    m.summarized("Built nothing yet. Walked east along the shore.", n)
    assert len(m.episodes) == MAX_EPISODES + 5 - 30 and m.episodes[0]["text"] == "event 30"
    assert Memory(tmp_path / "memory.json", clock).summary_text().startswith("Built nothing yet")
    assert not m.summary_due()


def test_the_thinker_writes_its_diary_with_one_model_call(tmp_path):
    t, game = thinker(tmp_path, chat=False), FakeGame()
    for i in range(MAX_EPISODES + 1):
        t.mem.add("misc", f"event {i}")
    transport, calls = model_transport(['{"goal":"x","do":["look_far N"]}'], delay=0.1, chat_line="Met nobody. Got wood.")

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=1.0 / 60)
    asyncio.run(go())
    diary = [c for c in calls if not c[1].get("stream")]
    assert len(diary) == 1 and "keeping your own diary" in diary[0][1]["messages"][0]["content"]
    assert t.mem.summary_text() == "Met nobody. Got wood."
    later = [c[1]["messages"][1]["content"] for c in calls if c[1].get("stream")][-1]
    assert "diary: Met nobody. Got wood." in later


def test_a_chat_recalls_who_hit_you_and_what_you_promised(tmp_path):
    t = thinker(tmp_path)
    clock = Clock()
    t.mem.now = clock
    t.mem.hit_by("Ana", 2, "sword", (0, 0), "day 1, 10:00", hp=16)
    clock.t += 600
    game = FakeGame(act_s=0.1, events=[{"type": "say", "from": "Ana", "text": "hey, remember me?"}])
    transport, calls = model_transport(['{"goal":"get wood","do":["dig N"]}'], delay=2.0,
                                       chat_line='{"say": "You hit me earlier. Why?", "do": "nothing", "promise": ""}')

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=1.0 / 60)
    asyncio.run(go())
    chat = next(c for c in calls if not c[1].get("stream"))[1]["messages"][1]["content"]
    assert "they hit you 1x" in chat and "10 min ago" in chat and "Ana says: hey, remember me?" in chat
    assert t.mem.people["Ana"]["you_said"] == ["You hit me earlier. Why?"]


def test_a_promise_is_kept_in_memory(tmp_path):
    t = thinker(tmp_path)
    game = FakeGame(act_s=0.1, events=[{"type": "say", "from": "Bo", "text": "build me a fence?"}])
    transport, _ = model_transport(['{"goal":"get wood","do":["dig N"]}'], delay=2.0,
                                   chat_line=json.dumps({"say": "Sure, after this tree.", "do": "nothing", "promise": "build Bo a fence"}))

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=1.0 / 60)
    asyncio.run(go())
    assert "you promised: build Bo a fence" in t.mem.person_text("Bo")


def test_danger_is_said_with_distance_what_it_does_and_the_health_trend():
    obs = fake_obs((0, 0))
    obs["entities"] = [{"id": 7, "kind": "tusker", "dx": 2, "dy": -1, "hp": 8}]
    obs["self"]["hp"] = 20
    body = Body()
    body.feel(obs, now=100.0)
    obs["self"]["hp"], obs["time"] = 14, "day 1, 21:00 (night)"
    obs["events"] = [{"type": "attack", "attacker": "a tusker", "target": "Vex", "damage": 3, "id": 7},
                     {"type": "windup", "id": 7, "kind": "tusker", "dir": [-1, 0]}]
    body.feel(obs, now=105.0)
    text = "\n".join(danger_lines(obs, body, now=106.0))
    assert "tusker #7 (2,-1) 3 tiles NE, hp 8, hits 3, a boar: charges in a straight line, step aside, drops meat" in text
    assert "WINDING UP A CHARGE toward W" in text
    assert "HURT: health 14/20 (-6 in 15 s), by a tusker." in text and "Night:" in text


def test_a_line_already_said_is_asked_again_and_a_plan_does_not_say_hi_five_times(tmp_path):
    t = thinker(tmp_path)
    t.mem.replied("Ana", "Busy with wood.")
    game = FakeGame(act_s=0.1, events=[{"type": "say", "from": "Ana", "text": "what are you doing?"}])
    transport, calls = model_transport(['{"goal":"get wood","do":["say hi","say hi","dig N"]}'], delay=0.2,
                                       chat_line='{"say": "Busy with wood.", "do": "nothing"}')

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=1.0 / 60)
    asyncio.run(go())
    chats = [c for c in calls if not c[1].get("stream")]
    assert len(chats) >= 2 and "You already said exactly that" in chats[1][1]["messages"][-1]["content"]
    said = [a[2]["text"] for a in game.acts if a[1] == "say"]
    assert "Busy with wood." not in said and said.count("hi") == 1 and t.stats["say_repeat"] >= 1
