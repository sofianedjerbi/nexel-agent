"""The learning env over the in-process sim (rl/core.py): shapes, determinism, reward accounting."""

import random

import numpy as np

from megastructure.rl.achieve import ACHIEVEMENTS, Ledger, crafter_score
from megastructure.rl.core import HURT, SimEnv
from megastructure.rl.encode import ACTIONS, N_ACTIONS, N_CHARS, SIDE, VEC, action_name


def idx(name):
    return next(i for i in range(N_ACTIONS) if action_name(i) == name)


def test_obs_shape():
    o = SimEnv(ep_ticks=200).reset(7)
    assert o["grid"].shape == (SIDE, SIDE) and o["grid"].dtype == np.uint8
    assert o["height"].shape == (SIDE, SIDE) and o["vec"].shape == (VEC,)
    assert o["grid"].max() < N_CHARS and o["grid"][SIDE // 2, SIDE // 2] != 0     # (the player, `@`)
    assert np.abs(o["height"]).max() <= 1


def rollout(seed, n=300):
    env, rng, out = SimEnv(ep_ticks=10 ** 6), random.Random(1), []
    env.reset(seed)
    for _ in range(n):
        o, r, d, t, _ = env.step(rng.randrange(N_ACTIONS))
        out.append((o["grid"].tobytes(), o["vec"].tobytes(), r, d))
    return out


def test_deterministic():
    assert rollout(11) == rollout(11)


def test_truncates_at_episode_end():
    env = SimEnv(ep_ticks=50)
    env.reset(3)
    for i in range(100):
        _, _, d, t, info = env.step(0)
        if t:
            assert info["ticks"] >= 50 and "achievements" in info
            return
    raise AssertionError("never truncated")


def test_first_time_only_and_crafts():
    env = SimEnv(ep_ticks=10 ** 6)
    env.reset(5)
    env.a.inventory.update({"wood": 12})
    rewards = [env.step(idx("craft_plank"))[1] for _ in range(8)]
    assert rewards.count(1.0) == 1 and sum(rewards) == 1.0          # craft_plank once (planks came from a craft: no collect)
    assert "craft_plank" in env.ledger.done and "collect_plank" not in env.ledger.done
    r = env.step(idx("craft_workbench"))[1] + sum(env.step(0)[1] for _ in range(3))
    assert r == 1.0 and env.ledger.counts["crafts"] >= 2


def during_next_tick(env, fn):
    """Something happens in the world during the next tick (as a pick-up or a blow would)."""
    step = env.sim.step

    def once():
        env.sim.step = step
        step()
        fn()
    env.sim.step = once


def test_collect_and_hurt_penalty():
    env = SimEnv(ep_ticks=10 ** 6)
    env.reset(5)
    during_next_tick(env, lambda: env.a.inventory.update(fiber=2))
    _, r, *_ = env.step(0)
    assert r == 1.0 and "collect_fiber" in env.ledger.done
    during_next_tick(env, lambda: env.sim.damage(env.a, 2, "test"))
    _, r, *_ = env.step(0)
    assert abs(r + 2 * HURT) < 1e-6 and env.ledger.counts["hits"] == 1


def test_death_ends_episode():
    env = SimEnv(ep_ticks=10 ** 6)
    env.reset(5)
    during_next_tick(env, lambda: env.sim.damage(env.a, 100, "test"))
    _, r, d, t, info = env.step(0)
    assert d and not t and info["died"] and r < 0


def test_failed_actions_counted():
    env = SimEnv(ep_ticks=10 ** 6)
    env.reset(5)
    env.step(idx("craft_sword"))
    env.step(0)
    assert env.ledger.counts["fails"] == 1


def test_ledger_and_score():
    L = Ledger()
    assert L.unlock("survive_night") == 1.0 and L.unlock("survive_night") == 0.0
    assert L.unlock("not_a_thing") == 0.0
    assert crafter_score({}) == 0.0
    assert abs(crafter_score({a: 1.0 for a in ACHIEVEMENTS}) - 100) < 1e-9
    assert len(set(ACTIONS)) == N_ACTIONS


def test_the_same_action_again_goes_on():
    env = SimEnv(ep_ticks=10 ** 6)
    env.reset(5)
    a = env.a
    env.sim.set_tile(a.x + 1, a.y, block="tree")
    for _ in range(40):                          # chopping by hand takes ~5 s: asked again each 0.5 s, it is not started over
        env.step(idx("dig_E"))
        if a.count("wood"):
            break
    assert a.count("wood") and "collect_wood" in env.ledger.done


def test_legal_mask_follows_the_rules():
    from megastructure.rl.encode import legal
    env = SimEnv(ep_ticks=10 ** 6)
    o = env.reset(5)
    assert o["mask"].shape == (N_ACTIONS,) and o["mask"][0]           # noop is always there
    env.a.inventory.update({"wood": 4, "stone": 3, "fiber": 1, "berries": 1})
    m = legal(env.sim.observe(env.a))
    assert m[idx("craft_workbench")] and m[idx("eat_berries")] and not m[idx("eat_meat")]
    assert not m[idx("craft_axe")]                                    # (no workbench within reach)
    assert not m[idx("dig_N_axe")] and m[idx("dig_N")]
