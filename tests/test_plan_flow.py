"""No dead time between plans, and nothing scripted in its place: the next call is always in flight, a failure keeps
the plan already streaming, the model's own "repeat" replays its plan, and a plan that keeps failing is a fact."""

import asyncio

import httpx

from megastructure.planner import Item
from test_thinker_realtime import FakeGame, model_transport, thinker


def play(t, game, plans, seconds, delay=0.1):
    transport, calls = model_transport(plans, delay=delay)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await t.play(game, http, minutes=seconds / 60)
    return asyncio.run(go()), calls


def test_the_model_repeat_plays_again_while_done_when_does_not_hold(tmp_path):
    t, game = thinker(tmp_path, chat=False), FakeGame(act_s=0.05)
    plan = '{"goal":"get wood","done_when":{"have":{"wood":99}},"repeat":4,"do":["look_far N"]}'
    s, calls = play(t, game, [plan], 3.5, delay=1.5)               # (slow calls: the repeats fill the wait)
    looks = [a for a in game.acts if a[1] == "look_far"]
    assert len(looks) >= 4 and t.stats["repeats"] >= 3


def test_a_failure_keeps_the_plan_already_streaming(tmp_path):
    t = thinker(tmp_path, chat=False)
    t.queue.append(Item("dig", {"dir": "N"}, "model", "", None, t.epoch, 1))
    hard = getattr(t, "hard_aborts", 0)
    t.abort("dig N failed", soft=True)
    assert getattr(t, "hard_aborts", 0) == hard                       # a streaming plan is not dropped for it
    t.abort("Ana hit you")
    assert t.hard_aborts == hard + 1


def test_the_next_call_is_in_flight_while_a_plan_still_has_commands(tmp_path):
    t, game = thinker(tmp_path, chat=False), FakeGame(act_s=0.6)
    plan = '{"goal":"look","do":["look_far N","look_far E","look_far S","look_far W"]}'
    s, calls = play(t, game, [plan], 3.0, delay=0.3)
    starts = [c[0] for c in calls if c[1].get("stream")]
    acts = [a[0] for a in game.acts]
    assert len(starts) >= 2 and any(acts[0] < c < acts[-1] for c in starts[1:])   # asked while the plan still ran
    assert s["idle_pct"] < 10


def test_a_plan_that_keeps_failing_is_said_as_a_fact(tmp_path):
    t, game = thinker(tmp_path, chat=False), FakeGame(fail={"dig N"})
    plan = '{"goal":"wood","do":["dig N"]}'
    s, calls = play(t, game, [plan], 4.0, delay=0.2)
    prompts = [c[1]["messages"][1]["content"] for c in calls if c[1].get("stream")]
    assert any("The plan 'dig N' failed" in p and "times in the last minute (last: dig N: nothing_there)" in p for p in prompts)
    assert not any(w in prompts[-1] for w in ("try something", "do something"))
