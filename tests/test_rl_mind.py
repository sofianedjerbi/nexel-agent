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
    assert "Your aims: bed (lacks" in again.http.sent[0]["messages"][1]["content"]


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
    from nexel_agent.rl.voice import places
    obs = {"self": {"x": 0, "y": 0, "spawn": [0, 20], "in_cave": False}, "grid": ["~T.", ".@."],
           "legend": {"~": "water", "T": "tree", ".": "grass", "@": "you"}}
    assert places(obs, {(5, -5): "mira's camp"}) == 'a sign "mira\'s camp" 7 tiles NE; your spawn point 20 tiles S; in sight: water, tree'
    assert places({"self": {"x": 0, "y": 0}, "grid": ["."], "legend": {".": "grass"}}, {}) == "none"


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

    async def burst():
        for text in ("wanna fight mossback", "?", "hello??"):
            v.hear("doben00", text, say)
        while v.answering:
            await asyncio.sleep(0.01)
    asyncio.run(burst())
    assert len(http.sent) == 1 and said == ["nah"]
    assert 'doben00 says to you: "wanna fight mossback" then "?" then "hello??"' in http.sent[0]["messages"][1]["content"]


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
