"""The gatsu0 livelock (2026-11-22): for an hour the model asked for a tree that was not in view; the plan walked
there, its digs were skipped, the filler walked off, the next identical plan walked back. Fake game, fake model."""

import asyncio
import json
import time

import httpx

from megastructure.planner import Item
from megastructure.plans import command_text
from test_thinker_realtime import FakeGame, fake_obs, model_transport, thinker


LIVELOCK = '{"goal":"harvest tree","do":["move_to 3 -1","dig N","dig E","dig W","dig S"]}'   # no tree there


def play(t, game, plans, seconds, delay=0.1):
    transport, calls = model_transport(plans, delay=delay)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await t.play(game, http, minutes=seconds / 60)
    return asyncio.run(go()), calls


def test_the_same_useless_plan_is_told_back_then_refused(tmp_path):
    """gatsu0 walked to (-55,-5) and back for an hour: the model kept asking for a tree that was not in view."""
    t, game = thinker(tmp_path, chat=False), FakeGame(act_s=0.05)
    game.tree = False
    s, calls = play(t, game, [LIVELOCK], 3.0)
    t.decision_log.flush()
    log = [json.loads(x) for x in (tmp_path / "Vex" / "decisions.jsonl").read_text().splitlines()]
    walks_there = [r for r in log if r.get("source") == "model" and r["tool"] == "move_to"]
    prompts = [c[1]["messages"][1]["content"] for c in calls if c[1].get("stream")]
    assert len(prompts) >= 6 and t.stats["plan_refused"] >= 3
    assert len(walks_there) <= 3                                         # (before: one walk there per plan, forever)
    told = [p for p in prompts if "changed nothing" in p]
    assert told and "nothing to dig there" in told[0], prompts[-1]
    assert all(p.count("your plan") <= 1 for p in prompts)             # said once, not four times over


def test_a_refused_plan_lets_the_filler_go_for_a_real_tree(tmp_path):
    t, game = thinker(tmp_path, chat=False), FakeGame(act_s=0.05)
    game.pos = [6, 0]                                                    # a real tree at (0,-2), 6 tiles off
    t.watch.nothing[("move_to 3 -1", "dig N", "dig E", "dig W", "dig S")] = 2
    play(t, game, [LIVELOCK], 1.5)
    assert ("dig", {"dir": "N"}) in [(a[1], a[2]) for a in game.acts]    # it got to the tree and dug it
    assert t.stats["plan_refused"] >= 1


def test_a_plan_asked_again_while_it_runs_goes_on_instead_of_restarting(tmp_path):
    t = thinker(tmp_path, chat=False)
    t.obs = fake_obs((0, 0))
    t.current = Item("move_to", {"x": 0, "y": -1}, "model", "wood", None, t.epoch)
    t.queue.append(Item("dig", {"dir": "N"}, "model", "wood", None, t.epoch))
    transport, _ = model_transport(['{"goal":"wood","do":["move_to 0 -1","dig N","craft plank"]}'], delay=0.01)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.plan_once(http)
    asyncio.run(go())
    assert [command_text(i.tool, i.args) for i in t.queue] == ["dig N", "craft plank"]
    assert t.stats["dup_dropped"] == 2


def test_the_filler_stays_where_the_plan_just_walked(tmp_path):
    t = thinker(tmp_path, chat=False)
    obs = fake_obs((5, 5), tree=False)
    t.anchor = (5, 5, time.monotonic() + 5)
    for _ in range(10):
        it = t.next_item(obs)
        assert it.source == "filler"
        if it.tool == "move_to":
            assert abs(it.args["x"] - 5) + abs(it.args["y"] - 5) <= 1
    t.anchor = None
    far = [t.next_item(obs) for _ in range(5)]
    assert any(i.tool == "move_to" and abs(i.args["x"] - 5) + abs(i.args["y"] - 5) > 2 for i in far)


def test_hungry_with_food_eats_at_8_without_dropping_the_plan(tmp_path):
    t = thinker(tmp_path, chat=False)
    obs = fake_obs((0, -1))
    obs["self"]["hunger"], obs["self"]["inventory"] = 5, {"egg": 12}
    t.queue.append(Item("dig", {"dir": "N"}, "model", "wood", None, t.epoch))
    it = t.next_item(obs)
    assert (it.tool, it.args, it.source) == ("eat", {"item": "egg"}, "reflex")
    assert [i.tool for i in t.queue] == ["dig"]                          # the plan goes on after the egg
    obs["self"]["inventory"] = {}
    assert "carry no food: get food first" in t.plan_messages(obs)[1]["content"]


def test_a_goal_whose_thing_is_not_in_view_is_told_so_and_the_filler_goes_looking(tmp_path):
    t = thinker(tmp_path, chat=False)
    obs = fake_obs((5, 5), tree=False)
    t.goal, t.anchor = "harvest tree", (5, 5, time.monotonic() + 5)
    assert "No tree anywhere in view" in t.plan_messages(obs)[1]["content"]
    picks = [t.next_item(obs) for _ in range(5)]
    assert all(i.tool == "move_to" and abs(i.args["x"] - 5) + abs(i.args["y"] - 5) >= 4 for i in picks)
    assert "No tree" not in t.plan_messages(fake_obs((0, 0)))[1]["content"]      # a tree in view: nothing said
