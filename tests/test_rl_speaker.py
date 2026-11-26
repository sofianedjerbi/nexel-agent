"""The one speaker (rl/voice.py Voice.speak): every line goes out through it, at one pace, with a record of the talk;
it answers what is said to it, not two others talking."""

import asyncio

from nexel_agent.rl import voice as V
from nexel_agent.rl.voice import Now, Voice

from test_rl_mind import FakeHTTP


def person(name, dx, dy=0):
    return {"kind": "agent", "name": name, "dx": dx, "dy": dy, "hp": 20}


def speaker(tmp_path, reply="sure", near=()):
    v = Voice("Rook", "quiet", "http://x", FakeHTTP(reply), tmp_path / "l.jsonl", Now("Rook", ("wood",)))
    v.now.see({"self": {"hp": 20, "hunger": 18, "inventory": {}}, "entities": list(near)})
    said = []

    async def out(line):
        said.append(line)
        return {"ok": True}
    v.out = out
    return v, said


async def settle(v):
    await asyncio.sleep(0)
    while v.answering:
        await asyncio.sleep(0.01)


def test_two_others_talking_get_no_reply_a_line_naming_it_does(tmp_path):
    v, said = speaker(tmp_path, near=(person("Mira", 3), person("doben00", 4)))

    async def go():
        v.hear("Mira", "doben want to go mining?")
        await settle(v)
        assert said == [] and v.http.sent == []
        v.hear("doben00", "hey rook, got wood?")
        await settle(v)
    asyncio.run(go())
    assert said == ["sure"] and len(v.http.sent) == 1


def test_the_only_one_near_is_talking_to_it(tmp_path):
    v, said = speaker(tmp_path, near=(person("Mira", 2),))

    async def go():
        v.hear("Mira", "got any wood?")
        await settle(v)
    asyncio.run(go())
    assert said == ["sure"]


def test_its_own_line_shows_in_the_next_reply_and_the_thread_goes_on(tmp_path):
    v, said = speaker(tmp_path, near=(person("Mira", 3), person("doben00", 4)))

    async def go():
        assert await v.speak("need 2 ingot, anyone?", "Mira") is not None      # (the planner's line)
        V.GAP_S = 0.0
        v.hear("Mira", "i have some")              # (not its name, two near: but it is in a talk with Mira)
        await settle(v)
    try:
        asyncio.run(go())
    finally:
        V.GAP_S = 2.0
    facts = v.http.sent[0]["messages"][1]["content"]
    assert 'Your talk with Mira before that: you: "need 2 ingot, anyone?"' in facts
    assert '"need 2 ingot, anyone?"' in facts.split("Your own last lines:")[1]
    assert said == ["need 2 ingot, anyone?", "sure"]


def test_one_pace_when_all_want_to_talk_at_once(tmp_path):
    v, said = speaker(tmp_path, near=(person("Mira", 2),))

    async def go():
        a = await v.speak("hey")                   # (the policy's say)
        b = await v.speak("wood anyone?")          # (the planner's line, just after: too soon, dropped)
        v.last_said -= V.GAP_S
        c = await v.speak("wood anyone?")          # (its own again, not spoken to since: the chat cooldown)
        v.heard_at = v.last_said + 0.1
        d = await v.speak("hey")                   # (spoken to since: may talk, but never the same line again)
        e = await v.speak("yo", reply=True)        # (an answer waits its turn, then goes)
        return a, b, c, d, e
    a, b, c, d, e = asyncio.run(go())
    assert a is not None and b is None and c is None and d is None and e is not None
    assert said == ["hey", "yo"]


def test_an_answer_to_what_was_said_since_goes_at_once(tmp_path):
    """#11: spoken to since its last line, its answer does not wait GAP_S after that line; an answer to nothing new does."""
    import time
    v, said = speaker(tmp_path, near=(person("Mira", 2),))

    async def go():
        await v.speak("hey")
        v.heard_at = time.monotonic()                      # (Mira spoke after its line)
        t0 = time.monotonic()
        a = await v.speak("sure", reply=True)
        t1 = time.monotonic()
        b = await v.speak("yo", reply=True)
        quick, slow = t1 - t0, time.monotonic() - t1
        return a, b, quick, slow
    a, b, quick, slow = asyncio.run(go())
    assert a is not None and quick < 0.5 and b is not None and slow >= V.GAP_S - 0.1, (quick, slow)
    assert said == ["hey", "sure", "yo"]


def test_a_claimed_deed_with_none_behind_it_goes_unsaid(tmp_path):
    """'I gave you the wood' with no give on record: the line is dropped; once the give is on record, it is said."""
    reply = '{"heard": ["chat"], "say": "I gave you the wood already", "promise": "", "deed": "gave wood"}'
    v, said = speaker(tmp_path, reply=reply, near=(person("Mira", 2),))

    async def go():
        v.hear("Mira", "hey Rook, where is my wood?")
        await settle(v)
        assert said == []
        v.now.remember("Mira", "you gave them wood")
        v.hear("Mira", "Rook? the wood?")
        await settle(v)
        assert said == ["I gave you the wood already"]
    asyncio.run(go())


def test_a_reply_never_waits_behind_a_thought_in_flight(tmp_path):
    """(#11) A thought is with the model when a line comes to answer: the reply goes out at once, beside it (the server
    decodes the two together), not after it; and no new thought starts while a reply is on its way."""
    gate, order = asyncio.Event(), []

    class Server:
        async def post(self, url, timeout, json):
            words = json["max_tokens"] == V.WORDS_TOKENS
            order.append("reply asked" if words else "thought asked")
            if not words:
                await gate.wait()                      # (a long thought: held until the reply is made)
            order.append("reply made" if words else "thought made")

            class R:
                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}], "usage": {}}
            return R()

    async def go():
        v = Voice("Rook", "a builder", "http://fake", Server(), tmp_path / "lat.jsonl")
        thought = asyncio.ensure_future(v.post([], V.THOUGHT_TOKENS, words=False))
        await asyncio.sleep(0)
        out, t = await asyncio.wait_for(v.post([], V.WORDS_TOKENS, words=True), 1)
        assert out == "ok" and t["queue_s"] < .1 and not thought.done()
        gate.set()
        await thought
    asyncio.run(go())
    assert order == ["thought asked", "reply asked", "reply made", "thought made"]
