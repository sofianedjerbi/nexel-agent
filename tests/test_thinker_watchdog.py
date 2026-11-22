"""The progress watchdog over every source, and the rate cap (gatsu0, 2026-11-22: 3,586 escape moves between
(-61,23) and (-60,23) in 15 min, all 'ok', the model never asked)."""

import asyncio
import time

from megastructure.watchdog import Watchdog, area
from test_thinker_livelock import play
from test_thinker_realtime import FakeGame, thinker
from test_thinker_trapped import canyon


def test_back_and_forth_with_nothing_gained_is_a_stall_and_that_routine_is_banned_here():
    dog, t, stall = Watchdog(), 0.0, None
    while stall is None and t < 30:
        stall = dog.record(t, -61 + int(t * 4) % 2, 23, {"dirt": 35}, "escape:bank")
        t += 0.25
    assert stall and stall["routine"] == "escape:bank" and 18 <= t <= 21
    assert "escape:bank" in dog.banned(-60, 23, t) and "escape:bank" not in dog.banned(-20, 23, t)
    assert dog.banned(-61, 23, t + 200) == set()                     # (bans run out)


def test_moving_on_or_gaining_something_is_progress():
    dog = Watchdog()
    assert not any(dog.record(i * 0.5, i // 4, 0, {}, "escape:swim") for i in range(80))      # 10 tiles in 40 s
    dog = Watchdog()
    assert not any(dog.record(i * 0.5, 0, 0, {"stone": i // 10}, "model") for i in range(80))  # digging in place
    assert area(-61, 23) == area(-57, 17)


class Shuttle(FakeGame):
    """The live fault: in the canyon, every move 'works' but the body only ever ends up on one of two tiles."""

    async def observe(self):
        obs = canyon()
        obs["self"]["x"], obs["self"]["y"] = self.pos
        return obs

    async def act(self, tool, **args):
        self.acts.append((time.monotonic(), tool, args))
        await asyncio.sleep(0.01)
        if tool == "move_to":
            self.pos = [-60 if self.pos[0] == -61 else -61, 23]
        return {"ok": True, "result": {"ticks": 1, "changed": {}}}


def test_the_two_tile_shuttle_is_caught_the_routine_changes_and_the_model_is_told(tmp_path):
    t, game = thinker(tmp_path, chat=False), Shuttle()
    game.pos = [-61, 23]
    t.dog = Watchdog(window=1.5)
    plan = '{"goal":"swim to shore","do":["look_far N"]}'
    s, calls = play(t, game, [plan], 6.0, delay=0.2)
    assert t.stats["stalls"] >= 1 and t.dog.stalls[0]["routine"] == "escape:swim"
    routines = [st["routine"] for st in t.dog.stalls]
    assert len(set(routines)) == len(routines)                       # never the same routine banned twice here
    assert len(routines) >= 2 and routines[1] == "escape:stair"      # swim banned here: on to the stair routine
    prompts = [c[1]["messages"][1]["content"] for c in calls if c[1].get("stream")]
    assert any("going back and forth at (-6" in p for p in prompts)


def test_never_more_than_about_three_actions_a_second(tmp_path):
    t, game = thinker(tmp_path, chat=False), Shuttle()              # every action instant
    game.pos = [-61, 23]
    play(t, game, ['{"goal":"x","do":["look_far N"]}'], 3.0)
    ts = [a[0] for a in game.acts]
    assert (len(ts) - 1) / (ts[-1] - ts[0]) <= 3.2
