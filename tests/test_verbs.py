"""Verbs (the how of one intent) in the in-process sim; trigger cards the model wrote, answering a hit in a fraction of a
second; goals checked against their own done_when. Fake model, no model server."""

import asyncio
import json
import math
import time

import httpx

from helpers import clear, make
from megastructure.cards import Cards, valid
from megastructure.goals import missing
from megastructure.sim.entities import Creature
from test_thinker_realtime import FakeGame, fake_obs, model_transport, thinker


class SimGame:
    """The in-process sim behind the client's async observe/act."""

    def __init__(self, sim, a):
        self.sim, self.a = sim, a

    async def observe(self):
        return self.sim.observe(self.a)

    async def act(self, tool, **args):
        act = self.sim.submit(self.a, tool, dict(args))
        for _ in range(900):
            if act.result is not None:              # (shaped as the server sends it)
                r = dict(act.result)
                if r.pop("ok", True):
                    return {"ok": True, "result": r}
                return {"ok": False, "error": r.get("error"), "detail": r.get("detail"), "result": r}
            self.sim.step()
        return {"ok": False, "error": "timeout"}


def setup(tmp_path):
    sim = make(tmp_path)
    a = sim.login("Vex", "tok")
    clear(sim, a)
    sim.keep_loaded = False
    return sim, a, SimGame(sim, a), thinker(tmp_path)


def test_dig_at_a_tree_walks_beside_it_and_digs_it(tmp_path):
    sim, a, game, t = setup(tmp_path)
    sim.world.set(a.x + 3, a.y + 1, block="tree")
    r = asyncio.run(t.verb(game, "dig_at", {"text": "tree"}))
    assert r.get("ok"), r
    assert a.inventory.get("wood", 0) >= 1 and sim.world.block(a.x - 0, a.y) == "none"


def test_move_away_gets_that_far_from_the_being(tmp_path):
    sim, a, game, t = setup(tmp_path)
    c = Creature(id=900, kind="cow", x=a.x + 2, y=a.y, hp=6)
    sim.creatures[c.id] = c
    r = asyncio.run(t.verb(game, "move_away", {"text": "cow 6"}))
    assert r.get("ok"), r
    assert math.dist((a.x, a.y), (c.x, c.y)) >= 5


def test_place_near_and_craft_here_walk_to_the_station(tmp_path):
    sim, a, game, t = setup(tmp_path)
    a.inventory.update({"workbench": 1, "wood": 2, "stone": 1})
    assert asyncio.run(t.verb(game, "place_near", {"text": "workbench"})).get("ok")
    ax, ay = a.x, a.y
    asyncio.run(game.act("move_to", x=ax + 5, y=ay))
    assert a.x == ax + 5
    r = asyncio.run(t.verb(game, "craft_here", {"text": "hoe"}))
    assert r.get("ok"), r
    assert a.inventory.get("hoe") == 1


def test_a_card_is_data_with_perception_facts_only():
    assert valid({"when": {"hit_by": "a tusker"}, "do": "move_away {id} 10"})["when"] == {"hit_by": "tusker"}
    assert valid({"when": {"mood": "angry"}, "do": ["attack {id}"]}) is None           # not a perception fact
    assert valid({"when": {"within": 3}, "do": []}) is None


def test_a_card_answers_a_hit_in_a_fraction_of_a_second(tmp_path):
    t = thinker(tmp_path)
    t.cards.add([{"when": {"hit_by": "any"}, "do": ["attack {id}"], "why": "nobody hits me twice"}])
    game = FakeGame(act_s=2.0)                               # a long dig is under way when the hit comes
    plan = '{"goal":"get wood","do":["dig N","dig N"]}'
    transport, _ = model_transport([plan], delay=0.1)
    hit_at = []

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            async def hit():
                await asyncio.sleep(1.5)
                hit_at.append(time.monotonic())
                game.events.append({"type": "attack", "attacker": "Ana", "target": "Vex", "damage": 2, "tick": 1})
            asyncio.create_task(hit())
            await t.play(game, http, minutes=3.0 / 60)
    orig = fake_obs

    async def observe():
        ev, game.events = game.events, []
        o = orig(tuple(game.pos), ev)
        o["entities"] = [{"id": 12, "kind": "agent", "name": "Ana", "dx": 1, "dy": 0, "hp": 20}]
        return o
    game.observe = observe
    asyncio.run(go())
    attacks = [a for a in game.acts if a[1] == "attack"]
    assert attacks and attacks[0][2] == {"target": 12}
    assert attacks[0][0] - hit_at[0] < 0.5                   # (the dig still had a second to run: it was replaced)
    logged = [json.loads(x) for x in open(tmp_path / "Vex" / "decisions.jsonl") if '"kind": "card"' in x]
    assert logged and logged[0]["reaction_s"] < 0.5 and t.stats["preempted"] >= 1


def test_on_joining_the_model_writes_its_first_cards_and_everyone_shares_them(tmp_path):
    t = thinker(tmp_path, cards=True)
    reply = json.dumps([{"when": {"being": "tusker", "within": 5}, "do": ["move_away {id} 10"], "why": "it charges"}])
    transport, calls = model_transport(['{"goal":"look","do":["look_far N"]}'], delay=0.2, chat_line=reply)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(FakeGame(), http, minutes=1.0 / 60)
    asyncio.run(go())
    ask = next(c for c in calls if not c[1].get("stream"))[1]["messages"]
    assert "reaction cards" in ask[0]["content"] and "tusker" in ask[1]["content"] and "joined" in ask[1]["content"]
    other = Cards(tmp_path / "_cards.json", "Ana")
    assert [c["do"] for _, c in other.usable()] == [["move_away {id} 10"]] and other.cards[0]["author"] == "Vex"


def test_a_goal_test_says_what_is_missing_and_the_body_sees_it_done(tmp_path):
    obs = fake_obs((0, 0))
    obs["self"]["inventory"] = {"wood": 2}
    assert missing({"have": {"wood": 4}, "hp_at_least": 15}, obs) == ["have 4 wood (you have 2)"]
    t, game = thinker(tmp_path), FakeGame()
    game.inv = {"wood": 2}
    plan = '{"goal":"get wood","done_when":{"have":{"wood":2}},"do":["dig N"]}'
    transport, calls = model_transport([plan], delay=0.1)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=1.5 / 60)
    asyncio.run(go())
    assert t.stats["goals_done"] >= 1 and any("goal 'get wood' done" in e["text"] for e in t.mem.episodes)
