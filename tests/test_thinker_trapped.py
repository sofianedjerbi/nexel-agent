"""Stuck, stalled, looping: the model is TOLD (gatsu0 2026-11-22: in a river under cliffs 10 layers high, every move_to
failed; 3,586 moves between two tiles in 15 min). Nothing scripted moves the body for it. Fake game, fake model."""

import asyncio
import time

import httpx

from nexel_agent.escape import climb_reach, stuck
from nexel_agent.watchdog import Watchdog
from test_thinker_realtime import FakeGame, model_transport, thinker


def play(t, game, plans, seconds, delay=0.1):
    transport, calls = model_transport(plans, delay=delay)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await t.play(game, http, minutes=seconds / 60)
    return asyncio.run(go()), calls


def canyon(bank=None, pit=False):
    """You at the centre of a 3-wide river running N-S (height 0) between cliffs at A (10). bank: (dx, dy) of a
    low bank (height 1) from there east to the edge of the view. pit: no water, a 1-tile hole 3 deep in flat land."""
    n, r = 21, 10
    grid = [["." for _ in range(n)] for _ in range(n)]
    hts = [["A" for _ in range(n)] for _ in range(n)]
    for j in range(n):
        for i in range(r - 1, r + 2):
            grid[j][i], hts[j][i] = ("." if pit else "~"), ("A" if pit else "0")
    if pit:
        hts = [["3"] * n for _ in range(n)]
        hts[r][r] = "0"
    if bank:
        for i in range(r + bank[0], n):
            grid[r + bank[1]][i], hts[r + bank[1]][i] = ".", "1"
    grid[r][r] = "@"
    return {"tick": 1, "time": "day 1, 09:00", "grid": ["".join(x) for x in grid], "heights": ["".join(x) for x in hts],
            "legend": {}, "entities": [], "items": [], "signs": [], "events": [],
            "self": {"id": 1, "name": "Vex", "x": 0, "y": 0, "z": 0, "hp": 14, "hunger": 9, "inventory": {},
                     "standing_on": "grass" if pit else "abyss", "swimming": not pit, "sheltered": False}}


def test_in_the_river_under_cliffs_it_knows_it_is_stuck():
    obs = canyon()
    assert stuck(obs, {})
    assert all(abs(q[0]) <= 1 for q in climb_reach(obs))            # only the water: the cliffs are too high
    assert not stuck(canyon(bank=(2, -4)), {})                       # a low bank that leads away
    assert stuck(canyon(pit=True), {"no_path": 3})


def test_the_prompt_says_it_is_stuck(tmp_path):
    t = thinker(tmp_path, chat=False)
    assert "No path leads from here to the land" in t.plan_messages(canyon())[1]["content"]
    assert "No path leads" not in t.plan_messages(canyon(bank=(2, -4)))[1]["content"]


def test_staying_put_with_nothing_gained_is_a_stall_and_moving_or_gaining_is_not():
    dog, t, stall = Watchdog(), 0.0, None
    while stall is None and t < 30:
        stall = dog.record(t, -61 + int(t * 4) % 2, 23, {"dirt": 35})
        t += 0.25
    assert stall and 18 <= t <= 21
    dog = Watchdog()
    assert not any(dog.record(i * 0.5, i // 4, 0, {}) for i in range(80))                  # 10 tiles in 40 s
    assert not any(dog.record(i * 0.5, 0, 0, {"stone": i // 10}) for i in range(80))       # digging in place


class Cliffs(FakeGame):
    """Every move_to fails (no_path)."""

    async def observe(self):
        obs = canyon()
        obs["self"]["x"], obs["self"]["y"] = self.pos
        return obs

    async def act(self, tool, **args):
        self.acts.append((time.monotonic(), tool, args))
        await asyncio.sleep(0.02)
        if tool == "move_to":
            return {"ok": False, "error": "no_path", "detail": "no walkable path"}
        return {"ok": True, "result": {"ticks": 1, "changed": {}}}


def test_trapped_the_model_hears_it_and_nothing_moves_on_its_own(tmp_path):
    t, game = thinker(tmp_path, chat=False), Cliffs()
    t.dog = Watchdog(window=1.0)
    s, calls = play(t, game, ['{"goal":"get out","do":["move_to 9 0","dig E"]}'], 3.0, delay=0.3)
    sent = [a for a in game.acts if a[1] == "move_to"]
    assert all(a[2] == {"x": 9, "y": 0} for a in sent)              # only what the model asked for
    same = [b[0] - a[0] for a, b in zip(sent, sent[1:])]
    assert all(dt >= 1.0 for dt in same)                             # the same failing move never within a second
    prompts = [c[1]["messages"][1]["content"] for c in calls if c[1].get("stream")]
    assert any("move_to 9 0 FAILED no_path" in p for p in prompts)
    assert any("No path leads" in p for p in prompts) and any("nothing in your pack changed" in p for p in prompts)


def test_never_more_than_about_three_actions_a_second(tmp_path):
    t, game = thinker(tmp_path, chat=False), FakeGame(act_s=0.0)
    play(t, game, ['{"goal":"x","do":["look_far N","look_far E","look_far S","look_far W"]}'], 3.0, delay=0.05)
    ts = [a[0] for a in game.acts]
    assert len(ts) > 3 and (len(ts) - 1) / (ts[-1] - ts[0]) <= 3.2
