"""Speech acts and promises (rl/acts.py, #20): the words model tells what each line heard does, so an answer that
names an item is not an ask (#9); a promise binds the body, and kept or broken goes on the record."""

import asyncio
import json

from nexel_agent.rl import acts
from nexel_agent.rl.goals import Goal
from nexel_agent.rl.mind import PROMISE, Mind
from nexel_agent.rl.now import Now
from nexel_agent.rl.social import SocialMemory
from nexel_agent.rl.voice import Voice

from test_rl_mind import FakeHTTP

ITEMS = ("wood", "stone", "ingot", "sword", "stone_block", "dirt", "fruit")


def test_the_reply_tells_each_line_and_the_promise():
    out = json.dumps({"heard": ["chat", "ask wood"], "say": "sure, 2 coming", "promise": "wood 2"})
    assert acts.read(out, 2, ITEMS) == ("sure, 2 coming", [("chat", None), ("ask", "wood")], ("wood", 2))
    assert acts.read("yo whats up", 1, ITEMS) == ("yo whats up", [], None)            # (no JSON: the line, as before)
    assert acts.read('{"heard": ["how swords"], "say": "", "promise": ""}', 1, ITEMS) == ("", [("how", "sword")], None)


def test_an_answer_or_a_statement_is_not_an_ask_only_an_ask_is():
    """#9: three lines name an item; the words model says which is an ask, and only that one opens."""
    now, social = Now("Rook", ITEMS), SocialMemory("Rook", ITEMS)
    now.words_tell = social.words_tell = True
    lines = [("Mira", "a sword is 2 ingot and 1 wood"), ("Mira", "i found stone over there"), ("Mira", "rook got any wood?")]
    social.see({"tick": 5, "self": {}, "events": [{"type": "say", "from": w, "text": t} for w, t in lines]})
    now.see({"self": {}, "events": [{"type": "say", "from": w, "text": t} for w, t in lines]})
    assert social.ask_open(5) is None and not now.asks                 # (a name in a line opens nothing)
    v = Voice("Rook", "", "http://x", FakeHTTP(json.dumps({"heard": ["chat", "chat", "ask wood"], "say": "sure"})),
              "/dev/null", now)
    v.on_heard = lambda who, kind, item: kind == "ask" and (social.heard_ask(who, item, 6), now.ask(who, "Rook", item))
    asyncio.run(v.line([(w, t, 0.0) for w, t in lines]))
    assert (social.ask_open(6), social.asked_by) == ("wood", "Mira") and now.asks == {("Mira", "Rook"): now.asks[("Mira", "Rook")]}
    assert now.asks[("Mira", "Rook")][0] == "wood"


def test_a_promise_is_the_goal_and_kept_or_broken_goes_on_the_record(tmp_path):
    m = Mind("Rook", "", "http://x", FakeHTTP("{}"), tmp_path / "memory.json", Now("Rook", ITEMS))
    m.names = {"mira": "Mira"}
    m.tick, m.pack = 100, {"wood": 3}
    m.commit("Mira", "wood", 2)
    assert (m.goal.kind, m.goal.item, m.goal.person, m.goal.promise, m.goal.until) == ("give", "wood", "Mira", True, 100 + PROMISE)
    obs = {"tick": 120, "self": {"inventory": {"wood": 2}}, "entities": [{"kind": "agent", "name": "Mira", "dx": 1, "dy": 0, "id": 7}]}
    m.track(obs, [("give", {"target": 7, "item": "wood", "n": 1}, {"ok": True, "to": "Mira"})], {})
    assert m.goal.kind == "free" and m.now.relations["Mira"]["record"]["you kept a promise to them"] == 1
    m.commit("Mira", "wood", 1)
    m.track({**obs, "tick": 120 + PROMISE + 1}, [], {})                 # (the deadline passed, nothing handed over)
    assert m.now.relations["Mira"]["record"]["you broke a promise to them"] == 1
    m.commit("Mira", "wood", 1)
    m.apply({"goal": "gather", "item": "stone"})                       # (the planner sets another goal: its word, broken)
    assert m.now.relations["Mira"]["record"]["you broke a promise to them"] == 2 and m.goal.kind == "gather"


def test_reading_out_the_pack_goes_unsaid_unless_asked():
    assert acts.pack_dump("stone block 26, dirt 13, still got fruit", ITEMS)
    assert acts.pack_dump("got 3 wood and 2 stone", ITEMS)
    assert not acts.pack_dump("got 3 wood, want some?", ITEMS) and not acts.pack_dump("stone near the hill", ITEMS)
    v = Voice("Rook", "", "http://x", FakeHTTP(""), "/dev/null", Now("Rook", ITEMS))
    said = []

    async def out(line):
        said.append(line)
        return {"ok": True}
    v.out = out
    assert asyncio.run(v.speak("stone_block 26, dirt 13", reply=True)) is None
    v.asked_pack = True                                                # (someone asked what it carries)
    assert asyncio.run(v.speak("stone_block 26, dirt 13", reply=True)) is not None and said == ["stone_block 26, dirt 13"]
