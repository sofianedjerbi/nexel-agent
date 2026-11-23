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


def test_facts_in_goal_and_notes_out(tmp_path):
    reply = ('{"goal": "stay", "with": "Mira", "item": "", "say": "sure, lets go", '
             '"relation_update": {"Mira": {"ally": true, "note": "gave me berries, asked to team up"}}}')
    http = FakeHTTP(reply)
    m = Mind("Rook", "a quiet player", "http://x", http, tmp_path / "memory.json")
    m.see(OBS)
    assert m.due()
    asyncio.run(m.think(OBS))
    facts = http.sent[0]["messages"][1]["content"]
    assert "Mira" in facts and "want to team up?" in facts and "gave you" in facts and "food 9/20" in facts
    assert (m.goal.kind, m.goal.person) == ("stay", "Mira") and m.to_say == "sure, lets go"
    again = Mind("Rook", "a quiet player", "http://x", FakeHTTP("{}"), tmp_path / "memory.json")
    assert again.relations["Mira"]["ally"] is True                 # (kept across sessions)
    assert json.loads((tmp_path / "memory.json").read_text())["Mira"]["notes"].startswith("gave me")


def test_a_bad_reply_keeps_the_goal(tmp_path):
    m = Mind("Rook", "", "http://x", FakeHTTP("no json here"), tmp_path / "m.json")
    asyncio.run(m.think(OBS))
    assert m.goal.kind == "free" and m.stats["errors"] == 1 and not m.busy


def test_voice_lines_timed_and_silence_kept(tmp_path):
    from nexel_agent.rl.voice import Voice
    me = {"hp": 20, "hunger": 18, "inventory": {"wood": 3}}
    v = Voice("Rook", "quiet", "http://x", FakeHTTP("sure, here"), tmp_path / "lat.jsonl")
    assert asyncio.run(v.line("Mira", "can i have wood?", me, "")) == "sure, here"
    v.http = FakeHTTP("nothing")
    assert asyncio.run(v.line("Mira", "hey", me, "")) == ""              # (it chose to say nothing)
    recs = [json.loads(x) for x in (tmp_path / "lat.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and "llm_s" in recs[0] and recs[0]["heard"] == "can i have wood?"


def test_places_are_only_real_ones():
    from nexel_agent.rl.voice import places
    obs = {"self": {"x": 0, "y": 0, "spawn": [0, 20], "in_cave": False}, "grid": ["~T.", ".@."],
           "legend": {"~": "water", "T": "tree", ".": "grass", "@": "you"}}
    assert places(obs, {(5, -5): "mira's camp"}) == 'a sign "mira\'s camp" 5 tiles NE; your spawn point 20 tiles S; in sight: water, tree'
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
