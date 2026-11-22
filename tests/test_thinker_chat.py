"""Talking to an AI player ("gatsu is full roleplaying not answering my messages"): it answers what was said, and does
what it was asked (come, stop), not one more riddle in character."""

import asyncio

import httpx
from test_thinker_realtime import FakeGame, model_transport, thinker

from megastructure.fastprompt import chat_messages, chat_parse


def test_the_reply_and_its_deed_are_read_apart():
    assert chat_parse("On my way, Doben!\nDO: come") == ("On my way, Doben!", "come")
    assert chat_parse('"Chopping wood by the river."\n**DO: nothing**') == ("Chopping wood by the river.", "")
    assert chat_parse("Fine, I'll wait.\nDO: STOP") == ("Fine, I'll wait.", "stop")
    assert chat_parse("<think>x</think>Hi!") == ("Hi!", "")


def test_the_prompt_asks_for_a_real_answer_and_says_where_you_are():
    m = chat_messages("gatsu0", "a grim swordsman", "harvest wood", "doben00", "where are you?", (),
                      {"x": 103, "y": 80, "hp": 20, "hunger": 18, "inventory": {"wood": 40}})
    assert "never cryptic" in m[0]["content"] and "DO: come" in m[0]["content"]
    assert "(103, 80)" in m[1]["content"] and "wood 40" in m[1]["content"] and "doben00 says: where are you?" in m[1]["content"]


def test_asked_to_come_it_answers_then_walks_to_who_asked(tmp_path):
    transport, calls = model_transport(['{"goal":"get wood","do":["dig N","dig N"]}'], delay=3.0, chat_line="Coming!\nDO: come")
    game = FakeGame(act_s=0.1, events=[{"type": "say", "from": "Ana", "text": "come here", "dx": 3, "dy": -2}])
    t = thinker(tmp_path)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=1.5 / 60)
    asyncio.run(go())
    i = next(i for i, a in enumerate(game.acts) if a[1] == "say")
    assert game.acts[i][2]["text"] == "Coming!"
    assert any(a[1] == "move_to" and (a[2]["x"], a[2]["y"]) == (3, -2) for a in game.acts[i + 1:]), game.acts
