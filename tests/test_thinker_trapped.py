"""gatsu0 trapped (2026-11-22): swimming in a river between cliffs 10 layers high, every move_to to the land
failed with no_path and the filler sent another one 3 times a second. Fake game, fake model."""

import asyncio
import time

from megastructure.escape import climb_reach, stuck, way_out
from megastructure.planner import Item
from test_thinker_realtime import FakeGame, model_transport, thinker
from test_thinker_livelock import play


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


def test_in_the_river_under_cliffs_it_knows_it_is_stuck_and_swims_on():
    obs, memory = canyon(), {}
    assert stuck(obs, memory)
    assert all(abs(q[0]) <= 1 for q in climb_reach(obs))            # only the water: the cliffs are too high
    tool, args, why, routine = way_out(obs, memory)
    assert (tool, routine) == ("move_to", "swim") and abs(args["x"]) <= 1 and abs(args["y"]) >= 8 and "swimming on" in why
    assert way_out(obs, memory)[1] == args                           # the same target until it gets there
    obs["self"]["y"] = args["y"]                                     # there (the river looks the same): on, same way
    again = way_out(obs, memory)[1]
    assert (again["y"] - args["y"] > 0) == (args["y"] > 0)           # keeps its heading: no back and forth


def test_a_low_bank_that_leads_away_is_the_way_out():
    obs = canyon(bank=(2, -4))                                       # a bank 1 layer up, going on east
    assert not stuck(obs, {})
    assert way_out(obs, {})[::3] == ("move_to", "bank") and way_out(obs, {})[1]["y"] == -4


def test_in_a_pit_it_cuts_a_step_and_jumps():
    obs, memory = canyon(pit=True), {"no_path": 3}
    assert stuck(obs, memory)
    assert way_out(obs, memory)[:2] == ("dig", {"dir": "up"})        # room over the head (nothing there: fine)
    tool, args, *_ = way_out(obs, memory)
    assert tool == "dig" and args["z"] == 2
    assert way_out(obs, memory)[:3] == ("jump", {"dir": args["dir"]}, "climbing out: onto the step")
    assert stuck(obs, {"climb_to": 3, **memory}) and memory["climb_to"] == 3   # on up to the wall's top


def test_swimming_banned_here_it_cuts_a_stair_out():
    obs, memory = canyon(), {}
    assert way_out(obs, memory)[3] == "swim"
    tool, args, why, _ = way_out(obs, memory, banned={"swim"})       # the watchdog found swimming went nowhere
    assert (tool, abs(args["x"]), why) == ("move_to", 1, "getting out: to the foot of the wall")
    obs["self"]["x"] = 1                                             # (the view stays the river's: same picture)
    shifted = canyon()
    shifted["grid"] = [(row[1:] + ".").replace("@", "~") for row in shifted["grid"]]
    shifted["grid"][10] = shifted["grid"][10][:10] + "@" + shifted["grid"][10][11:]
    shifted["heights"] = [row[1:] + "A" for row in shifted["heights"]]
    assert way_out(shifted, memory, {"swim"})[:2] == ("dig", {"dir": "up"}) and memory["climb_to"] == 10


def test_the_prompt_says_it_is_stuck(tmp_path):
    t = thinker(tmp_path, chat=False)
    assert "You are stuck: no path from here" in t.plan_messages(canyon())[1]["content"]
    assert "You are stuck" not in t.plan_messages(canyon(bank=(2, -4)))[1]["content"]


class Cliffs(FakeGame):
    """Every move_to fails (no_path) except along the river; counts what was sent."""

    async def observe(self):
        obs = canyon()
        obs["self"]["x"], obs["self"]["y"] = self.pos
        return obs

    async def act(self, tool, **args):
        self.acts.append((time.monotonic(), tool, args))
        await asyncio.sleep(0.02)
        if tool == "move_to" and abs(args["x"] - self.pos[0]) > 1:
            self.fails.append((time.monotonic(), tool, args))
            return {"ok": False, "error": "no_path", "detail": "no walkable path"}
        if tool == "move_to":
            self.pos = [args["x"], args["y"]]
        return {"ok": True, "result": {"ticks": 1, "changed": {}}}


def test_trapped_it_backs_off_then_escapes_instead_of_hammering(tmp_path):
    t, game = thinker(tmp_path, chat=False), Cliffs()
    game.fails = []
    plan = '{"goal":"get pickaxe","do":["move_to 9 0","dig E"]}'      # the model wants the cliff top
    s, _ = play(t, game, [plan], 2.0, delay=0.3)
    fails = game.fails
    same = [b[0] - a[0] for a, b in zip(fails, fails[1:]) if a[2] == b[2]]
    assert all(dt >= 1.0 for dt in same)                             # never the same failing move within a second
    assert t.stats["escape:ok"] >= 1 and game.pos[1] != 0            # it swam along the river
    assert len(fails) <= 4                                           # (before: ~3 failing sends a second, forever)
