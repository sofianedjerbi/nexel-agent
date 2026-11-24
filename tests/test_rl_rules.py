"""The RL env's legal mask, live-seed starts, social achievements and goals."""

import random

import numpy as np

from nexel_agent.rl.achieve import ACHIEVEMENTS, Ledger, crafter_score
from nexel_agent.rl.core import HURT, SimEnv
from rl_kit import during_next_tick, idx, neighbour_beside, to_command_for
from nexel_agent.rl.encode import ACTIONS, N_ACTIONS, N_CHARS, SIDE, VEC, action_name, legal


def test_legal_mask_follows_the_rules():
    from nexel_agent.rl.encode import legal
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    o = env.reset(5)
    assert o["mask"].shape == (N_ACTIONS,) and o["mask"][0]           # noop is always there
    env.a.inventory.update({"wood": 4, "stone": 3, "fiber": 1, "berries": 1})
    env.a.hunger = 10                                                 # (hungry: food can be eaten)
    m = legal(env.sim.observe(env.a))
    assert m[idx("craft_workbench")] and m[idx("eat_berries")] and not m[idx("eat_meat")]
    assert not m[idx("craft_axe")]                                    # (no workbench within reach)
    assert not m[idx("dig_N_axe")] and m[idx("dig_N")]
    if not env.sim.observe(env.a).get("heights"):                     # (flat all round: nothing in front that high)
        assert not m[idx("dig_at_N_1")] and not m[idx("dig_at_N_2")]


def test_live_seed_share_and_roaming_start():
    env = SimEnv(ep_ticks=100, live_seed=12, live_pct=100)
    env.reset(3)
    assert env.sim.seed == 12 and env.a.online
    spawn = SimEnv(ep_ticks=100, roam=0, rivals=0)
    spawn.reset(3)
    roam = SimEnv(ep_ticks=100, rivals=0)
    roam.reset(3)
    assert (spawn.a.x, spawn.a.y) != (roam.a.x, roam.a.y) and roam.rival is None


def test_social_once_each_and_helping():
    from nexel_agent.rl.social import SOCIAL
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a, f = env.sim, env.a, env.folk[0]
    f.next = sim.tick + 1
    env.step(0)
    assert f.on
    f.act = 10 ** 9                                   # (it stands still and does nothing more by itself)
    f.a.x, f.a.y, f.a.z = a.x + 1, a.y, 99
    a.inventory["wood"] = 3
    sim.submit(f.a, "say", {"text": "got any wood to spare?"})
    o, *_ = env.step(0)
    from nexel_agent.rl.encode import ACTIONS_V1, BODY, ITEMS, KINDS, N_NEAR
    s = BODY + len(ITEMS) + N_NEAR * (3 + len(KINDS)) + ACTIONS_V1
    assert o["vec"][s] == 1 and o["vec"][s + 1] == 1    # (spoken to; asked for something it carries)
    env.step(idx("give"))
    env.step(0)
    assert {"gave_item", "helped"} <= env.ledger.done and f.a.count("wood") >= 1
    env.step(idx("say"))
    env.step(0)
    assert "answered_someone" in env.ledger.done
    before = set(env.ledger.done) & set(SOCIAL)
    for _ in range(4):                                # (talking more earns nothing more)
        env.step(idx("say"))
    assert set(env.ledger.done) & set(SOCIAL) <= before | {"greeted_someone"} and len(SOCIAL) == 6


def test_live_body_imports():
    import nexel_agent.rl.live as live
    assert live.LiveBody and live.split


def test_refusals_the_obs_shows_are_masked():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    env.reset(5)
    a = env.a
    env.sim.set_tile(a.x + 1, a.y, block="none")
    a.inventory.update({"hoe": 1, "berries": 1})
    a.hunger = 20
    m = legal(env.sim.observe(a), "E")
    assert not m[idx("dig_E")] and not m[idx("dig_E_axe")] and m[idx("dig_E_hoe")]    # (bare level ground: tilling)
    assert not m[idx("eat_berries")]                                                   # (full)
    env.sim.set_tile(a.x + 1, a.y, block="tree")
    m = legal(env.sim.observe(a), "E")
    assert m[idx("dig_E")] and not m[idx("dig_E_hoe")] and not m[idx("step_E")]


def test_goals_pay_once_when_done():
    from nexel_agent.rl.goals import Goal
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    env.reset(5)
    env.goals.goal, env.goals.next = Goal("craft", "plank", None, env.sim.tick), env.sim.tick + 10 ** 6
    env.a.inventory["wood"] = 3
    _, r, *_ = env.step(idx("craft_plank"))
    assert r >= 2.0 and env.goals.counts["craft"][0] == 1 and env.goals.goal.kind == "free"   # (achievement + goal)
    _, r, *_ = env.step(idx("craft_plank"))
    assert r < 1.0                                                                   # (nothing more for the same)


def test_a_goal_about_a_person_it_struck_is_not_paid():
    from nexel_agent.rl.goals import Goal
    for strike, paid in ((False, 1), (True, 0)):
        env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, shadow_pct=0)
        env.reset(5)
        f = neighbour_beside(env)
        env.goals.goal, env.goals.next = Goal("avoid", None, f.a.name, env.sim.tick), env.sim.tick + 10 ** 6
        if strike:
            env.step(idx("attack"))
            env.step(0)
        f.a.x, f.a.y = env.a.x + 40, env.a.y + 40             # (far off for a minute: the goal's own measure of done)
        f.act = 10 ** 9
        for _ in range(140):
            env.step(0)
            f.a.x, f.a.y = env.a.x + 40, env.a.y + 40
        assert env.goals.counts["avoid"][0] == paid


def test_ask_goal_and_a_neighbour_who_says_yes():
    from nexel_agent.rl.goals import Goal
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a, f = env.sim, env.a, env.folk[0]
    f.next = sim.tick + 1
    env.step(0)
    f.a.x, f.a.y, f.a.z = a.x + 1, a.y, 99
    f.rng.random = lambda: 0.0                        # (it says yes)
    env.goals.goal, env.goals.next = Goal("ask", "wood", f.a.name, sim.tick), sim.tick + 10 ** 6
    env.obs = sim.observe(a)
    assert env.encoded()["mask"][idx("ask")]
    env.step(idx("ask"))
    for _ in range(30):
        env.step(0)
        if env.goals.counts["ask"][0]:
            break
    assert env.goals.counts["ask"][0] == 1 and a.count("wood") >= 1
