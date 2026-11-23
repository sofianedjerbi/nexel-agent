"""Talking to an AI player ("gatsu is full roleplaying not answering my messages"): it answers what was said, and what it
means to do about it goes to its mind (the next plan), never to a scripted move."""

import asyncio

import httpx
from test_thinker_realtime import FakeGame, model_transport, thinker

from nexel_agent.chat import chat_messages
from nexel_agent.fastprompt import chat_parse


def test_the_reply_and_its_deed_are_read_apart():
    assert chat_parse("On my way, Doben!\nDO: come") == ("On my way, Doben!", "follow")
    assert chat_parse('"Chopping wood by the river."\n**DO: nothing**') == ("Chopping wood by the river.", "")
    assert chat_parse("Fine, I'll wait.\nDO: STOP") == ("Fine, I'll wait.", "stop")
    assert chat_parse("<think>x</think>Hi!") == ("Hi!", "")


def test_the_prompt_asks_for_a_real_answer_and_says_where_you_are():
    m = chat_messages("gatsu0", "a grim swordsman", "harvest wood", "doben00", "where are you?", (),
                      {"x": 103, "y": 80, "hp": 20, "hunger": 18, "inventory": {"wood": 40}})
    assert "never cryptic" in m[0]["content"] and "mostly lowercase" in m[0]["content"] and "not needy" in m[0]["content"] and "no coordinates" in m[0]["content"] and "\"intent\"" in m[0]["content"]
    assert "(103, 80)" in m[1]["content"] and "wood 40" in m[1]["content"] and "doben00 says: where are you?" in m[1]["content"]


def test_asked_to_come_it_answers_and_the_next_plan_knows_it_means_to_follow(tmp_path):
    transport, calls = model_transport(['{"goal":"get wood","do":["dig N","dig N"]}'], delay=1.0,
                                       chat_line='{"say": "Coming!", "intent": "follow"}')
    game = FakeGame(act_s=0.1, events=[{"type": "say", "from": "Ana", "text": "come here", "dx": 3, "dy": -2}])
    t = thinker(tmp_path)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=2.5 / 60)
    asyncio.run(go())
    assert [a[2]["text"] for a in game.acts if a[1] == "say"] == ["Coming!"]
    assert not any(a[1] == "move_to" and (a[2]["x"], a[2]["y"]) == (3, -2) for a in game.acts)   # no scripted walk
    plans = [c[1]["messages"][-1]["content"] for c in calls if c[1].get("stream")]
    assert any('you said "Coming!", meaning to follow (Ana is at (3,-2))' in p for p in plans)


def test_the_json_reply_and_a_name_prefix():
    assert chat_parse('{"say": "Hey! Chopping wood.", "do": "nothing"}') == ("Hey! Chopping wood.", "")
    assert chat_parse('{"say": "On my way.", "intent": "follow"}') == ("On my way.", "follow")
    assert chat_parse('{"say": "Back off.", "intent": "fight"}') == ("Back off.", "fight")
    assert chat_parse("Gatsu0: Hey, I'm harvesting wood. DO: come") == ("Hey, I'm harvesting wood.", "follow")


def test_a_hit_is_told_to_the_planner_not_answered_by_a_script(tmp_path):
    """"do not hardcode response logic ... hit him he should hit back or run away": a hit stops the plan and the next
    plan call hears who hit, with what, and how hurt you are; no scripted line is said."""
    transport, calls = model_transport(['{"goal":"get wood","do":["dig N"]}'], delay=0.2, chat_line="no")
    game = FakeGame(act_s=0.1, events=[{"type": "attack", "attacker": "Ana", "target": "Vex", "damage": 2, "weapon": "sword", "dx": 1, "dy": 0}])
    t = thinker(tmp_path)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=1.2 / 60)
    asyncio.run(go())
    assert not [c for c in calls if not c[1].get("stream")] and not [a for a in game.acts if a[1] == "say"]
    plans = [c[1]["messages"][-1]["content"] for c in calls if c[1].get("stream")]
    assert any("Ana just hit you with sword: -2 health" in p for p in plans), plans[-1][-600:]
