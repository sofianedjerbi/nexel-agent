"""The LLM layer over the learned body (rl/mind.py): facts in, a goal and notes out, notes kept across sessions."""

import asyncio
import json

from nexel_agent.rl.mind import Mind


class FakeHTTP:
    def __init__(self, reply):
        self.reply, self.sent = reply, []

    async def post(self, url, timeout, json):
        self.sent.append(json)
        reply = self.reply

        class R:
            def json(self):
                return {"choices": [{"message": {"content": reply}}]}
        return R()


OBS = {"tick": 100, "time": "day 1, 09:00", "self": {"hp": 14, "hunger": 9, "inventory": {"wood": 2}},
       "entities": [{"kind": "agent", "name": "Mira", "dx": 2, "dy": 1, "hp": 20, "holding": "nothing"}],
       "events": [{"type": "say", "from": "Mira", "text": "want to team up?"},
                  {"type": "give", "from": "Mira", "to": "Rook", "items": {"berries": 1}}]}


def test_facts_in_goal_aims_and_allies_out(tmp_path):
    reply = ('{"goal": "stay", "with": "Mira", "item": "", "say": "sure, lets go", "aims": ["bed", "door"], '
             '"relation_update": {"Mira": {"ally": true, "note": "she seems nice"}}}')
    http = FakeHTTP(reply)
    m = Mind("Rook", "a quiet player", "http://x", http, tmp_path / "memory.json")
    m.now.see(OBS)
    m.see(OBS)
    assert m.due()
    asyncio.run(m.think(OBS))
    facts = http.sent[0]["messages"][1]["content"]
    assert "Mira" in facts and "want to team up?" in facts and "gave you berries" in facts and "food 9/20" in facts
    assert "no aims yet" in facts
    assert (m.goal.kind, m.goal.person) == ("stay", "Mira") and m.to_say == "sure, lets go"
    again = Mind("Rook", "a quiet player", "http://x", FakeHTTP("{}"), tmp_path / "memory.json")
    assert again.relations["Mira"]["ally"] is True                 # (kept across sessions)
    mem = json.loads((tmp_path / "memory.json").read_text())
    assert mem["aims"] == ["bed", "door"] and mem["people"]["Mira"]["record"] == {"gave you berries": 1}   # (no model notes)
    assert again.aims == ["bed", "door"]
    asyncio.run(again.think(OBS))
    assert "Your aims: bed (you have 0; made from" in again.http.sent[0]["messages"][1]["content"]


def test_a_bad_reply_keeps_the_goal(tmp_path):
    m = Mind("Rook", "", "http://x", FakeHTTP("no json here"), tmp_path / "m.json")
    asyncio.run(m.think(OBS))
    assert m.goal.kind == "free" and m.stats["errors"] == 1 and not m.busy


def test_voice_lines_timed_and_silence_kept(tmp_path):
    from nexel_agent.rl.voice import Now, Voice
    v = Voice("Rook", "quiet", "http://x", FakeHTTP("sure, here"), tmp_path / "lat.jsonl", Now("Rook", ("wood",)))
    v.now.see({"self": {"hp": 20, "hunger": 18, "inventory": {"wood": 3}}})
    assert asyncio.run(v.line([("Mira", "can i have wood?", 0.0)])) == "sure, here"
    assert "You carry 3 wood." in v.http.sent[0]["messages"][1]["content"]
    v.http = FakeHTTP("nothing")
    assert asyncio.run(v.line([("Mira", "hey", 0.0)])) == ""              # (it chose to say nothing)
    recs = [json.loads(x) for x in (tmp_path / "lat.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and "llm_s" in recs[0] and recs[0]["heard"] == "can i have wood?"


def test_places_are_only_real_ones():
    from nexel_agent.rl.now import places
    obs = {"self": {"x": 0, "y": 0, "spawn": [0, 20], "in_cave": False}, "grid": ["~T.", ".@."],
           "legend": {"~": "water", "T": "tree", ".": "grass", "@": "you"}}
    assert places(obs, {(5, -5): "mira's camp"}) == 'a sign "mira\'s camp" 7 tiles NE; your spawn point 20 tiles S; in sight: water, tree'
    assert places({"self": {"x": 0, "y": 0}, "grid": ["."], "legend": {".": "grass"}}, {}) == "none"
    # (#24) the ore in the rock that shows, from observe's ores list, nearest first, each once (the ore block "*" is gone)
    ores = [{"dx": 1, "dy": 0, "z": -2, "ore": "iron_ore"}, {"dx": 2, "dy": 0, "z": -2, "ore": "coal"}, {"dx": 3, "dy": 1, "z": -3, "ore": "iron_ore"}]
    assert places({"self": {"x": 0, "y": 0}, "grid": ["M"], "legend": {"M": "rock"}, "ores": ores}, {}) == "in sight: rock, iron ore, coal"


def test_chat_lines_go_before_thoughts(tmp_path):
    from nexel_agent.rl.voice import Voice
    order = []

    class Slow(FakeHTTP):
        async def post(self, url, timeout, json):
            order.append(json["max_tokens"])
            await asyncio.sleep(0.05)
            return await super().post(url, timeout, json)

    v = Voice("Rook", "", "http://x", Slow("ok"), tmp_path / "l.jsonl")

    async def both():
        v.words_waiting += 1                                            # (a chat line is waiting)
        thought = asyncio.create_task(v.post([], 100, words=False))
        await asyncio.sleep(0.02)
        v.words_waiting -= 1
        await v.post([], 30, words=True)
        await thought
    asyncio.run(both())
    assert order[0] == 30


def fight(now):
    """Three sporelings at its side, Mira who just hit it, its health going 20 -> 17."""
    me = {"x": 0, "y": 0, "hp": 20, "hunger": 18, "inventory": {"wood": 3}}
    now.see({"self": me, "time": "day 2, 10:00", "entities": []})
    now.did("climb_N")
    now.did("attack_back")
    now.see({"self": {**me, "hp": 17, "inventory": {"wood": 3, "stone": 1}}, "time": "day 2, 10:01",
             "entities": [{"kind": "sporeling", "id": i, "dx": dx, "dy": 0, "hp": 1} for i, dx in ((1, 1), (2, -1), (3, 1))]
             + [{"kind": "agent", "name": "Mira", "dx": 0, "dy": 1, "hp": 20, "holding": "axe"}],
             "events": [{"type": "attack", "attacker": "Mira", "target": "Rook", "damage": 2},
                        {"type": "attack", "attacker": "a sporeling", "target": "Rook", "damage": 0.5},
                        {"type": "attack", "attacker": "a sporeling", "target": "Rook", "damage": 0.5},
                        {"type": "attack", "attacker": "Rook", "target": "sporeling#9", "damage": 1},
                        {"type": "say", "from": "Mira", "text": "lol"}]})


def test_the_words_and_the_planner_see_the_same_fight(tmp_path):
    from nexel_agent.rl.voice import Now, Voice
    now = Now("Rook")
    fight(now)
    s = now.text()
    for bit in ("health 17/20 (was 20", "3 sporeling (foe) 1 tiles away", "Mira right here, health 20, holding axe",
                "Mira hit you", "a sporeling hit you x2", "you hit a sporeling", "climb north, attack back",
                "you got 1 stone", 'Mira: "lol"'):
        assert bit in s, (bit, s)
    vh, mh = FakeHTTP("ow"), FakeHTTP("{}")
    v = Voice("Rook", "", "http://x", vh, tmp_path / "l.jsonl", now=now)
    m = Mind("Rook", "", "http://x", mh, tmp_path / "m.json", now=now)
    asyncio.run(v.line([]))
    asyncio.run(m.think(now.obs))
    assert s in vh.sent[0]["messages"][1]["content"] and s in mh.sent[0]["messages"][1]["content"]


def test_a_burst_of_lines_gets_one_reply(tmp_path):
    from nexel_agent.rl.voice import Voice
    http, said = FakeHTTP("nah"), []
    v = Voice("Rook", "", "http://x", http, tmp_path / "l.jsonl")

    async def say(line):
        said.append(line)
        return {"ok": True}
    v.out = say

    async def burst():
        for text in ("rook wanna fight mossback", "?", "hello??"):
            v.hear("doben00", text)
        while v.answering:
            await asyncio.sleep(0.01)
    asyncio.run(burst())
    assert len(http.sent) == 1 and said == ["nah"]
    assert 'doben00 says to you: "rook wanna fight mossback" then "?" then "hello??"' in http.sent[0]["messages"][1]["content"]


def test_records_are_outcomes_only(monkeypatch):
    import nexel_agent.rl.voice as voice
    clock = [0.0]
    monkeypatch.setattr(voice.time, "monotonic", lambda: clock[0])
    now = voice.Now("Rook", ("wood", "stone"))
    mira = {"kind": "agent", "name": "Mira", "dx": 1, "dy": 0, "hp": 20}
    now.see({"self": {"hp": 20}, "entities": [mira], "events": [{"type": "say", "from": "Mira", "text": "got stone?"}]})
    now.ask("Rook", None, "wood")
    now.see({"self": {"hp": 20}, "entities": [mira], "events": [{"type": "give", "from": "Rook", "to": "Mira", "items": {"stone": 2}}]})
    clock[0] = 31.0
    now.see({"self": {"hp": 20}, "entities": [mira], "events": []})
    assert now.relations["Mira"]["record"] == {"asked you for stone": 1, "you asked them for wood": 1, "you gave them stone": 2,
                                               "did not give you the wood you asked for": 1}


def test_what_lies_about_and_the_pack_are_facts():
    from nexel_agent.rl.voice import Now
    now = Now("Rook")
    pack = {f"item{i}": 20 - i for i in range(12)} | {"wood": 1}
    now.see({"self": {"hp": 20, "hunger": 20, "inventory": pack},
             "items": [{"dx": 0, "dy": 5, "items": {"wood": 3}}, {"dx": 2, "dy": 0, "items": {"meat": 2, "fiber": 1}}]})
    s = now.text()
    assert "2 meat, 1 fiber on the ground 2 tiles E; 3 wood on the ground 5 tiles S" in s       # (nearest first)
    assert "You carry 20 item0, 19 item1" in s and "9 item11 and 1 more kind." in s             # (the PACK_KINDS biggest: #11)
    now.see({"self": {"hp": 20, "hunger": 20, "inventory": {"stone": 3, "wood": 1}}})
    assert "You carry 3 stone, 1 wood." in now.text()                                          # (a single wood counts)
    now.see({"self": {"hp": 20, "hunger": 20, "inventory": {}}})
    assert "You carry nothing." in now.text() and "nothing on the ground" in now.text()


def test_what_a_thing_takes_is_the_games_own_recipe():
    from nexel_worlds.tiles import RECIPES
    from nexel_agent.rl.facts import counted, needs
    sword = RECIPES["sword"]
    made = next(k for k in sword.inputs if k in RECIPES)          # (the bar it takes, whatever the game calls it)
    ingot = RECIPES[made]
    obs = {"self": {"inventory": {"wood": 5}, "stations": []}}
    s = needs("sword", obs)
    assert s.startswith(f"you have 0; made from {counted(sword.inputs)} at a {sword.station}; you lack ")
    assert f"({made} is made from {counted(ingot.inputs)} at a {ingot.station})" in s
    have = {k: 9 for k in sword.inputs}
    assert needs("sword", {"self": {"inventory": have, "stations": []}}).endswith(f"no {sword.station} in reach")
    assert needs("sword", {"self": {"inventory": have, "stations": [sword.station]}}).endswith("you can make it now")
    assert needs("wood", obs, ("wood",)) == "you have 5, no recipe makes it"
    assert needs("good_hoe", obs, ("wood",)).endswith("no item you know goes by that name")


def test_what_someone_told_it_is_kept(tmp_path):
    reply = ('{"goal": "free", "aims": ["good_hoe"], "learned": {"good hoe": {"by": "mira", "what": "just a hoe, 2 wood 1 stone"},'
             ' "sword": {"by": "Nobody", "what": "made up"}}}')
    m = Mind("Rook", "", "http://x", FakeHTTP(reply), tmp_path / "memory.json")
    m.now.see(OBS)
    asyncio.run(m.think(OBS))
    assert m.told == {"good_hoe": 'Mira: "just a hoe, 2 wood 1 stone"'}      # (from someone it saw only)
    again = Mind("Rook", "", "http://x", FakeHTTP("{}"), tmp_path / "memory.json")
    assert "told" not in again.relations and again.told == m.told
    asyncio.run(again.think(OBS))
    facts = again.http.sent[0]["messages"][1]["content"]
    assert 'What people told you: good_hoe (Mira: "just a hoe, 2 wood 1 stone").' in facts
    assert "Your aims: good_hoe (you have 0, no recipe makes it" in facts


def test_a_goal_is_judged_live_as_in_the_sim_and_ends(tmp_path):
    """The planner's goal is judged by the sim's own judge (rl/goals.py): done, or timed out; then it is free again,
    the planner thinks at once and is told how it went. A goal for an item the game has not ends at once."""
    from nexel_agent.rl.goals import TIMEOUT
    from nexel_agent.rl.voice import Now
    m = Mind("Rook", "calm", "http://x", None, tmp_path / "memory.json", now=Now("Rook", ("wood", "hoe")))
    obs = {"tick": 100, "self": {"inventory": {}}, "entities": [{"kind": "agent", "name": "Mira", "id": 7, "dx": 4, "dy": 0}]}
    m.see(obs)
    m.names = {"mira": "Mira"}
    m.apply({"goal": "ask", "with": "Mira", "item": "wood"})
    m.track(obs, [("say", {"ask": "wood"}, {"ok": True})], {})
    assert m.goal.kind == "ask" and m.goal.asked == 1 and "you asked 1 times" in m.facts(obs)
    m.track({**obs, "tick": 101}, [], {"Mira": {"wood": 2}})
    assert m.goal.kind == "free" and m.salient and "ask wood (with Mira): " in m.ended and "done" in m.ended
    m.apply({"goal": "ask", "with": "Mira", "item": "hoe"})
    m.track({**obs, "tick": 101 + TIMEOUT, "entities": []}, [], {})
    assert m.goal.kind == "free" and "timed out" in m.ended and "now out of sight" in m.ended
    m.apply({"goal": "ask", "with": "Mira", "item": "good_hoe"})
    assert m.goal.kind == "free" and "no item called good_hoe" in m.ended and "Your last goal" in m.facts(obs)


def test_an_aim_said_as_a_phrase_is_the_items_it_names(tmp_path):
    """Live, the model wrote aims as phrases ("a good hoe"): stored as such, no recipe matched them and it asked a
    neighbour for a bed 95 times out of 95. An aim is the game's items it names."""
    reply = '{"goal": "free", "aims": ["a house with a door and bed", "a good hoe", "a bed"]}'
    m = Mind("Rook", "a quiet player", "http://x", FakeHTTP(reply), tmp_path / "memory.json")
    m.now.items = ("bed", "door", "hoe", "stone")
    m.now.see(OBS)
    m.see(OBS)
    asyncio.run(m.think(OBS))
    assert m.aims == ["door", "bed", "hoe"]


def test_an_aim_it_owns_reads_as_owned_and_an_unseen_input_names_what_gives_it(tmp_path):
    """Live trial 4: two beds carried, 'bed: you lack 3 fiber' read for an hour (15 fiber gathers, no build), and
    'nobody you know had any' was all it was told of fiber. An owned aim lacks nothing; an input with no source in
    sight names what in the world gives it."""
    m = Mind("Rook", "a loud builder", "http://x", FakeHTTP("{}"), tmp_path / "memory.json")
    m.aims = ["bed", "axe"]
    obs = {**OBS, "self": {**OBS["self"], "inventory": {"bed": 2, "wood": 2, "stone": 2}}, "grid": ["....", "...."]}
    m.now.see(obs)
    facts = m.messages(obs)[1]["content"]
    assert "bed (you have 2: it is yours already)" in facts
    assert "1 fiber (nobody you know had any; none in sight: " in facts and "thicket" in facts
    assert "3 fiber" not in facts


def test_its_own_ask_line_is_on_the_record_and_a_tree_in_sight_gives_wood(tmp_path):
    """Live trial 4: 25 lines asking Mira for wood, none recorded; the goal read 'you have not asked yet' for 96 plans,
    a tree in sight all along. Its line to the person of an ask goal is the ask; the facts name what in sight gives
    an item its aims lack."""
    reply = '{"goal": "ask", "with": "Mira", "item": "wood", "say": "got wood?", "aims": ["door"]}'
    m = Mind("Rook", "a quiet player", "http://x", FakeHTTP(reply), tmp_path / "memory.json")
    m.now.items = ("wood", "door", "plank")
    obs = {**OBS, "self": {**OBS["self"], "inventory": {}}, "grid": ["....", ".T..", "...."]}
    m.now.see(obs)
    m.see(obs)
    asyncio.run(m.think(obs))
    assert "in sight, tree gives it" in m.messages(obs)[1]["content"]

    class Voice:
        async def speak(self, line, to):
            return {"ok": True}
    asyncio.run(m.speak(Voice(), "got wood?"))
    assert m.goal.asked == 1 and ("Rook", "Mira") in m.now.asks


def test_the_facts_tell_how_its_goals_went(tmp_path):
    from nexel_agent.rl.goals import Goal
    m = Mind("Rook", "a quiet player", "http://x", FakeHTTP("{}"), tmp_path / "memory.json")
    for _ in range(3):
        m.end(Goal("ask", "wood", "Mira"), "timed out, not done")
    m.end(Goal("gather", "wood"), "done")
    m.now.see(OBS)
    assert "ask wood (with Mira): 0 of 3; gather wood: 1 of 1" in m.messages(OBS)[1]["content"]


def test_near_you_names_the_nearest_kinds_and_a_few_piles():
    from nexel_agent.rl.now import NEAR_KINDS, PILES
    from nexel_agent.rl.voice import Now
    now = Now("Rook")
    beings = [{"kind": f"beast{i}", "dx": i + 1, "dy": 0} for i in range(NEAR_KINDS + 2)]
    piles = [{"dx": 0, "dy": i + 1, "items": {"wood": 1}} for i in range(PILES + 2)]
    now.see({"self": {"hp": 20, "hunger": 20, "inventory": {}}, "entities": beings, "items": piles})
    s = now.text()
    assert "beast0" in s and f"beast{NEAR_KINDS - 1}" in s and f"beast{NEAR_KINDS}" not in s, s     # (the nearest kinds, #11)
    assert "2 more kinds of creature further" in s and s.count("on the ground") == PILES, s


def test_a_build_goal_steers_the_body_to_build_till_nothing_is_left():
    """The planner says build (its home): the body's action becomes build, but not when hurt, and the building ends
    when build has nothing left to lay (told in the next facts)."""
    import numpy as np
    from nexel_agent.rl.actions import ACTIONS
    b = next(k for k, a in enumerate(ACTIONS) if a[0] == "build")
    m = Mind("Rook", "a quiet player", "http://x", FakeHTTP('{"goal": "build", "aims": ["bed"]}'), tmp_path_factory())
    m.now.see(OBS)
    m.see(OBS)
    asyncio.run(m.think(OBS))
    assert m.building and m.goal.kind == "free"
    mask = np.ones(len(ACTIONS), bool)
    assert m.steer(3, mask, 0.0) == b and m.steer(3, mask, 2.0) == 3       # (hurt: the body's own)
    mask[b] = False
    assert m.steer(3, mask, 0.0) == 3 and not m.building and "nothing more" in m.ended


def tmp_path_factory():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp()) / "memory.json"


def test_a_goal_left_for_another_is_set_aside_and_told_and_kept():
    """The owner: change goals whenever needed, but remember - after the fight it still wants the fiber. A goal left
    before it is done goes on the agenda, told in the facts; set again, it comes off; it is kept across sessions."""
    from nexel_agent.rl.goals import Goal
    path = tmp_path_factory()
    m = Mind("Rook", "a quiet player", "http://x", FakeHTTP("{}"), path)
    m.now.items = ("fiber", "wood")
    m.goal = Goal("ask", "fiber", "doben00", m.tick, 0)
    m.names = {"doben00": "doben00"}                                         # (doben00 is there: seen)
    m.apply({"goal": "stay", "with": "doben00"})
    m.now.see(OBS)
    facts = m.messages(OBS)[1]["content"]
    assert "Goals you set aside" in facts and "ask fiber (with doben00)" in facts
    again = Mind("Rook", "a quiet player", "http://x", FakeHTTP("{}"), path)
    assert again.agenda.saved() and again.agenda.saved()[0]["item"] == "fiber"
    again.names = {"doben00": "doben00"}
    again.apply({"goal": "ask", "with": "doben00", "item": "fiber"})
    assert not any(x["item"] == "fiber" for x in again.agenda.saved())


def test_a_goal_with_someone_it_never_saw_is_not_set_and_it_is_told_so():
    """Live trial 2, alone in the world: the planner asked "player_near_forest_edge" for fiber 19 times - a player it
    made up, its goal then telling the name back to it. A goal with a person it never saw ends at once, told why."""
    path = tmp_path_factory()
    m = Mind("Rook", "a quiet player", "http://x", FakeHTTP("{}"), path)
    m.now.items = ("fiber", "wood")
    m.apply({"goal": "ask", "with": "player_near_forest_edge", "item": "fiber"})
    assert m.goal.person is None and "nobody named player_near_forest_edge" in m.ended
    m.names = {"mira": "Mira"}
    m.apply({"goal": "ask", "with": "mira", "item": "fiber"})
    assert m.goal.person == "Mira"                                            # (seen: as the game spells her)
    m.apply({"goal": "gather", "with": "nobody_here", "item": "wood"})
    assert (m.goal.kind, m.goal.person) == ("gather", None)                  # (a goal with no one: it stands)
