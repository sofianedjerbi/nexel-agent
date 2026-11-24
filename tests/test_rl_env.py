"""The learning env over the in-process sim (rl/core.py): shapes, determinism, reward accounting."""

import random

import numpy as np

from nexel_agent.rl.achieve import ACHIEVEMENTS, Ledger, crafter_score
from nexel_agent.rl.core import HURT, SimEnv
from rl_kit import during_next_tick, idx, neighbour_beside, to_command_for
from nexel_agent.rl.encode import ACTIONS, N_ACTIONS, N_CHARS, SIDE, VEC, action_name, legal


def test_obs_shape():
    o = SimEnv(ep_ticks=200).reset(7)
    assert o["grid"].shape == (SIDE, SIDE) and o["grid"].dtype == np.uint8
    assert o["height"].shape == (SIDE, SIDE) and o["vec"].shape == (VEC,)
    assert o["grid"].max() < N_CHARS and o["grid"][SIDE // 2, SIDE // 2] != 0     # (the player, `@`)
    assert np.abs(o["height"]).max() <= 1


def rollout(seed, n=300):
    env, rng, out = SimEnv(ep_ticks=10 ** 6, roam=0), random.Random(1), []
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
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    env.a.inventory.update({"wood": 12})
    rewards = [env.step(idx("craft_plank"))[1] for _ in range(8)]
    assert rewards.count(1.0) == 1 and sum(rewards) == 1.0          # craft_plank once (planks came from a craft: no collect)
    assert "craft_plank" in env.ledger.done and "collect_plank" not in env.ledger.done
    r = env.step(idx("craft_workbench"))[1] + sum(env.step(0)[1] for _ in range(3))
    assert r == 1.0 and env.ledger.counts["crafts"] >= 2


def test_collect_and_hurt_penalty():
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    during_next_tick(env, lambda: env.a.inventory.update(fiber=2))
    _, r, *_ = env.step(0)
    assert r == 1.0 and "collect_fiber" in env.ledger.done
    during_next_tick(env, lambda: env.sim.damage(env.a, 2, "test"))
    _, r, *_ = env.step(0)
    assert abs(r + 2 * HURT) < 1e-6 and env.ledger.counts["hits"] == 1
    during_next_tick(env, lambda: setattr(env.a, "hp", env.a.hp + 2))
    _, r, *_ = env.step(0)
    assert r == 0.0                                             # (healing does not pay the loss back)


def test_a_start_one_step_from_a_tool():
    from megastructure.tiles import RECIPES
    from nexel_agent.rl.starts import TOOLS
    plain, benches, ready, kinds = SimEnv(ep_ticks=100, rivals=0, folk=0), 0, 0, set()
    for seed in range(30):
        env = SimEnv(ep_ticks=100, rivals=0, folk=0, gear_pct=100)
        env.reset(seed)
        plain.reset(seed)
        assert (env.a.x, env.a.y, env.a.hunger) == (plain.a.x, plain.a.y, plain.a.hunger)      # (the same start otherwise)
        assert set(env.a.inventory) - set(plain.a.inventory) <= {"wood", "stone", "fiber"}
        benches += env.sim.near_block(env.a, "workbench", 2) and not plain.sim.near_block(plain.a, "workbench", 2)
        can = [t for t in TOOLS if all(env.a.count(k) >= n for k, n in RECIPES[t].inputs.items())]
        ready += bool(can)
        kinds |= set(can)
    assert 6 <= benches <= 24 and 12 <= ready < 30 and len(kinds) == 3       # (varied: a bench or not, all there or not)


def test_an_episode_rarely_starts_full():
    food = []
    for seed in range(12):
        env = SimEnv(ep_ticks=100, roam=0, rivals=0, folk=0)
        env.reset(seed)
        food.append(env.a.hunger)
    assert min(food) >= 8 and max(food) <= 20 and len(set(food)) > 3 and sum(f < 20 for f in food) >= 8


def test_death_ends_episode():
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    during_next_tick(env, lambda: env.sim.damage(env.a, 100, "test"))
    _, r, d, t, info = env.step(0)
    assert d and not t and info["died"] and r < 0


def test_failed_actions_counted():
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
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
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    a = env.a
    env.sim.set_tile(a.x + 1, a.y, block="tree")
    for _ in range(40):                          # chopping by hand takes ~5 s: asked again each 0.5 s, it is not started over
        env.step(idx("dig_E"))
        if a.count("wood"):
            break
    assert a.count("wood") and "collect_wood" in env.ledger.done


def test_an_action_runs_till_it_ends():
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    a = env.a
    env.sim.set_tile(a.x + 1, a.y, block="tree")
    t0 = env.sim.tick
    _, r, *_ = env.step(idx("dig_E"))             # one decision: the chop (~2.4 s by hand) goes to its end
    assert a.count("wood") and r >= 1.0 and env.sim.tick - t0 >= 20


def test_a_pit_is_not_a_trap():
    """Dug down three layers, the actions can cut steps up out of it (dig the land in front at head height, jump)."""
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(1000003)
    sim, a = env.sim, env.a

    def until_done(name, n=60):
        for _ in range(n):
            env.step(idx(name))
            if env.running and env.running[1].result is not None:
                return env.running[1].result
        return env.running[1].result

    for _ in range(3):
        assert until_done("dig_down")["ok"]
    low = sim.height(a)
    assert env.encoded()["mask"][idx("dig_at_N_2")]                  # (legal in a pit: land in front, that high)
    for _ in range(3):
        until_done("dig_at_N_2")
        until_done("jump_N")
    assert sim.height(a) > low


def test_farming_and_depth_milestones():
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    a = env.a
    env.sim.set_tile(a.x + 1, a.y, block="none")
    a.inventory.update({"hoe": 1, "seed": 2, "watering_can": 1})
    env.obs = env.sim.observe(a)
    for name in ("dig_E_hoe", "place_seed", "water"):
        assert env.encoded()["mask"][idx(name)]
        env.step(idx(name))
        env.step(0)
    assert {"till", "plant", "water_crop", "place_seed"} <= env.ledger.done, env.ledger.done
    for _ in range(6):
        env.step(idx("dig_down"))
    assert env.sim.height(a) <= -5 and "depth_5" not in env.ledger.done      # (counted once back up)
    for _ in range(40):
        env.step(idx("climb_N") if env.encoded()["mask"][idx("climb_N")] else idx("step_N"))
        if env.sim.height(a) >= 0 and not env.sim.in_cave(a):
            break
    env.step(0)
    assert "depth_5" in env.ledger.done, env.sim.height(a)


def test_climb_out_of_a_deep_pit():
    """Ten layers down, the same action (climb N) asked again and again brings it back up."""
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(1000003)
    sim, a = env.sim, env.a
    for _ in range(10):
        env.step(idx("dig_down"))
    low = sim.height(a)
    assert low <= -8
    for _ in range(30):
        if env.encoded()["mask"][idx("climb_N")]:
            env.step(idx("climb_N"))
        else:
            env.step(idx("step_N"))
    assert sim.height(a) >= low + 6, (low, sim.height(a))


def test_a_sparring_partner_comes_and_can_be_beaten():
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    env.rival.next = env.sim.tick + 1
    env.step(0)
    assert env.rival.on and env.rival.dist() <= 7
    hp = env.a.hp
    for _ in range(80):
        m = env.encoded()["mask"]
        env.step(idx("attack") if m[idx("attack")] else 0)
        if "defeat_player" in env.ledger.done:
            break
    assert "defeat_player" in env.ledger.done and not env.rival.on
    assert env.a.hp < hp or env.ledger.counts["hits"] >= 0


def test_one_attack_is_a_whole_fight():
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    sim, a, rv = env.sim, env.a, env.rival
    rv.next = sim.tick + 1
    env.step(0)
    rv.a.x, rv.a.y, rv.a.z = a.x + 1, a.y, 99          # (right beside: in reach)
    env.social.hit_me[rv.a.name] = sim.tick            # (it has just hit the player)
    env.obs = sim.observe(a)
    t0 = sim.tick
    env.step(idx("attack"))                            # one decision: swings till it falls (a fight back)
    env.step(0)
    assert "defeat_player" in env.ledger.done and sim.tick - t0 <= 60, sim.tick - t0


def test_a_sparring_partner_spares_you():
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    env.rival.next = env.sim.tick + 1
    for _ in range(200):
        _, _, died, _, _ = env.step(0)
        assert not died
        if env.ledger.done & {"survive_attack", "escape_attack"}:
            break
    assert "survive_attack" in env.ledger.done
