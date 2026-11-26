"""The mind's side of a home (rl/home.py, rl/mind.py; #30): kept across sessions and handed back to the body, told in
one line of facts from the judge and the blueprint, and an aim the planner can name. A fake model throughout."""

import asyncio
import json
from types import SimpleNamespace

from test_rl_mind import FakeHTTP

from nexel_agent.rl import home
from nexel_agent.rl.build import PLAN
from nexel_agent.rl.core import SimEnv
from nexel_agent.rl.mind import Mind

PLAN_JSON = '{"goal": "free", "aims": ["a home with a bed"]}'


def hut(laid):
    """A sim world and its player, the hut's blueprint (rl/build.py PLAN) laid from the inside's corner one SE of it,
    the pieces `laid` of it (walls, the door, the furniture; never the roof). -> env, home, obs"""
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    hx, hy = a.x + 1, a.y + 1
    for i in range(-3, 7):
        for j in range(-3, 7):
            sim.world.set(hx + i, hy + j, ground="grass", block="none")
    for dx, dy, p in PLAN:
        if p != "roof" and p in laid:
            sim.world.set(hx + dx, hy + dy, block={"wall": "wood_wall", "door": "door"}.get(p, p))
    a.inventory.clear()
    a.inventory.update({"wood": 2})
    return env, (hx, hy), sim.observe(a)


def test_its_home_is_told_from_the_judge_and_what_it_lacks():
    _, at, obs = hut({"wall", "door", "campfire"})
    line = home.told(obs, at)
    assert line.startswith("Your home right here: closed, a door, no roof (0% covered), lit, no bed"), line
    assert "it makes no sense yet: no chest" in line and "it does not look good:" in line and "no roof" in line
    lack = home.lacks(obs, at)                                 # (the roof, the bed: in wood and fiber, the 2 wood carried off)
    assert lack and lack.get("wood", 0) > 0 and "fiber" in lack and line.endswith(f"it lacks {', '.join(f'{n:g} {k}' for k, n in lack.items())}.")
    _, at, obs = hut(set())                                    # nothing laid: not closed, and all of it lacking
    assert home.told(obs, at).startswith("Your home right here: not closed") and home.lacks(obs, at)["wood"] > lack["wood"]


def test_the_home_is_kept_across_sessions_and_handed_back(tmp_path):
    body = SimpleNamespace(home=(12, -4))
    m = Mind("Rook", "", "http://x", FakeHTTP(PLAN_JSON), tmp_path / "memory.json")
    m.social = body
    m.see({"tick": 1, "self": {"x": 0, "y": 0, "inventory": {}}})
    assert m.home == (12, -4)
    asyncio.run(m.think({"tick": 2, "self": {"x": 0, "y": 0, "hp": 20, "hunger": 20, "inventory": {}}, "grid": [], "legend": {}}))
    assert json.loads((tmp_path / "memory.json").read_text())["home"] == [12, -4]
    body2 = SimpleNamespace(home=None)                                # the next session: a body with no home yet
    again = Mind("Rook", "", "http://x", FakeHTTP("{}"), tmp_path / "memory.json")
    again.social = body2
    again.see({"tick": 1, "self": {"x": 0, "y": 0, "inventory": {}}})
    assert body2.home == (12, -4) and again.home == (12, -4)          # handed back
    body2.home = None                                                 # the body gives it up (no room there): kept so
    again.see({"tick": 2, "self": {"x": 0, "y": 0, "inventory": {}}})
    assert body2.home is None and again.home is None


def test_the_planner_can_aim_at_its_home_and_sees_what_it_lacks(tmp_path):
    env, at, obs = hut({"wall", "door"})
    m = Mind("Rook", "", "http://x", FakeHTTP(PLAN_JSON), tmp_path / "memory.json")
    m.social, m.now.items = SimpleNamespace(home=at), {"bed", "door", "wood_wall"}
    m.see(obs)
    asyncio.run(m.think(obs))
    assert m.aims == ["home", "bed"], m.aims
    facts = m.facts(obs)
    assert f"home (it lacks {', '.join(f'{n:g} {k}' for k, n in home.lacks(obs, at).items())})" in facts
    assert "Your home right here: closed" in facts


def test_a_players_day_night_falls_and_its_home_and_bed(tmp_path):
    """(#15) From five minutes before night: how long until night (or dawn), where its home is, its bed there."""
    from nexel_worlds.sim.constants import DAY_TICKS, TICK_HZ
    _, at, obs = hut({"wall", "door", "campfire"})
    day = DAY_TICKS * 10
    obs = {**obs, "tick": day + int(.3 * DAY_TICKS)}                       # morning: nothing to say
    assert home.day(obs, at) is None
    obs["tick"] = day + int(.75 * DAY_TICKS) - 2 * 60 * TICK_HZ           # two minutes before night
    assert home.day(obs, at) == ("Night falls in 2 min: glooms come out; your home is right here, with no bed yet: aim for one, "
                                 "and sleep in it (a bed is where you wake after dying).")
    obs["tick"] = day + int(.9 * DAY_TICKS)
    assert home.day(obs, at).startswith("It is night: glooms are out, dawn in 2 min")
    assert home.day(obs, None).endswith("you have no home to go to: a bed is where you wake after dying.")
    _, at, obs = hut({"wall", "door", "campfire", "bed"})
    obs["tick"] = day + int(.9 * DAY_TICKS)
    assert "with a bed (" in home.day(obs, at)
    m = Mind("Rook", "", "http://x", FakeHTTP("{}"), tmp_path / "memory.json")
    m.social = SimpleNamespace(home=at)
    m.see(obs)
    assert "It is night: glooms are out" in m.facts(obs)
