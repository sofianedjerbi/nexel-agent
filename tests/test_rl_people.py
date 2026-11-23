"""The RL env's neighbours and rivals: blows, grudges, packs, attack slots; live-like land and rooms."""

import random

import numpy as np

from nexel_agent.rl.achieve import ACHIEVEMENTS, Ledger, crafter_score
from nexel_agent.rl.core import HURT, SimEnv
from rl_kit import during_next_tick, idx, neighbour_beside, to_command_for
from nexel_agent.rl.encode import ACTIONS, N_ACTIONS, N_CHARS, SIDE, VEC, action_name, legal


def test_a_neighbour_hit_hits_back_and_goes():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    f = neighbour_beside(env)
    hp = env.a.hp
    o, *_ = env.step(idx("attack"))
    from nexel_agent.rl.encode import PEOPLE
    late = N_ACTIONS - 181
    assert o["vec"][VEC - late - PEOPLE + 1] == 1               # (the nearest: "I hit it lately")
    f.act = 0
    for _ in range(40):
        env.step(0)
    assert env.a.hp < hp and not f.on                          # (it hit back, then went)


def test_the_chat_cooldown():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    neighbour_beside(env)
    assert env.encoded()["mask"][idx("say")]
    env.step(idx("say"))
    env.step(0)
    assert not env.encoded()["mask"][idx("say")]                # (just spoke, nobody answered: not for 30 s)
    env.sim.submit(env.folk[0].a, "say", {"text": "hey"})
    env.step(0)
    assert env.encoded()["mask"][idx("say")]                    # (spoken to: a line back is fine)


def test_a_kill_costs_the_pack_and_the_game_goes_on():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, folk=0, death_ends=0)
    env.reset(5)
    sim, a, rv = env.sim, env.a, env.rival
    a.inventory.update({"wood": 5})
    rv.next = sim.tick + 1
    env.step(0)
    rv.a.x, rv.a.y, rv.a.z = a.x + 1, a.y, 99
    a.hp = 1
    for _ in range(20):
        _, _, end, _, _ = env.step(0)
        assert not end
        if a.deaths:
            break
    env.step(0)
    assert a.deaths == 1 and not a.count("wood") and rv.a.count("wood") == 5 and env.ledger.counts["deaths"] == 1


def test_neighbours_remember_a_blow():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    f = neighbour_beside(env)
    env.step(idx("attack"))
    env.step(0)
    assert f.grudge
    f.angry = 0
    f.a.events = [{"type": "say", "from": env.a.name, "text": "could you spare some wood?"}]
    f.a.inventory["wood"] = 2
    f.tick()
    assert f.asked is None                                      # (it says no)


def test_a_fallen_neighbours_pack_is_not_collected():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    f = neighbour_beside(env)
    f.a.inventory.clear()
    f.a.inventory["fiber"] = 2
    f.a.hp = 1
    env.step(idx("attack"))
    for _ in range(20):
        env.step(idx("pick_up") if env.encoded()["mask"][idx("pick_up")] else 0)
        if env.a.count("fiber"):
            break
    env.step(0)
    assert env.a.count("fiber") == 2 and "collect_fiber" not in env.ledger.done
    assert env.ledger.counts["hit_neighbour_unprovoked"] == 1


def test_a_grudge_fades():
    from nexel_agent.rl.folk import GRUDGE, GIFT_EASES
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    f = env.folk[0]
    f.grudge = True
    assert f.grudge
    env.sim.tick += GRUDGE - GIFT_EASES + 1
    assert f.grudge
    f.grudge_until -= GIFT_EASES                                # (as a gift does)
    assert not f.grudge


def test_attack_back_goes_for_the_hitter():
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    sim, a, rv = env.sim, env.a, env.rival
    f = neighbour_beside(env)                                   # (the nearest: the neighbour)
    rv.next = sim.tick + 1
    env.step(0)
    rv.a.x, rv.a.y, rv.a.z = a.x - 2, a.y, 99
    sim.submit(rv.a, "attack", {"target": a.id})
    env.step(0)
    env.step(0)
    o = env.obs
    assert env.encoded()["mask"][idx("attack_back")]
    cmd = to_command_for(env, "attack_back")
    assert cmd[1]["target"] == rv.a.id and o["entities"][0]["id"] == f.a.id


def test_a_first_blow_is_one_swing_a_fight_back_goes_on():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, shadow_pct=0)
    env.reset(5)
    f = neighbour_beside(env)
    cmd = to_command_for(env, "attack")
    assert cmd[1]["target"] == f.a.id and not cmd[1].get("fight")
    from nexel_agent.rl.core import split
    assert len(split(cmd)) == 1
    env.social.hit_me[f.a.name] = env.sim.tick                 # (it hit me just now)
    cmd = to_command_for(env, "attack")
    assert cmd[1].get("fight") and len(split(cmd)) > 1


def test_a_shadowing_neighbour_comes_at_once():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, shadow_pct=100)
    env.reset(5)
    for _ in range(40):
        env.step(0)
    f = env.folk[0]
    assert f.shadowing and f.on and f.dist() <= 3


def test_live_like_land_and_fighting_armed():
    from collections import Counter
    lush, thin = SimEnv(ep_ticks=100, rivals=0, folk=0), SimEnv(ep_ticks=100, rivals=0, folk=0, thin_pct=100)
    counts = []
    for env in (lush, thin):
        env.reset(1000001)
        a = env.a
        counts.append(Counter(env.sim.world.block(a.x + dx, a.y + dy) for dx in range(-40, 41) for dy in range(-40, 41)))
    assert counts[1]["tree"] < counts[0]["tree"] / 2 and counts[1]["thicket"] >= 3
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, shadow_pct=0)
    env.reset(5)
    f = neighbour_beside(env)
    env.a.inventory["spear"] = 1
    env.obs = env.sim.observe(env.a)
    env.step(idx("attack"))
    env.step(0)
    assert "fought_armed" in env.ledger.done


def test_a_room_it_built():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    env.reset(5)
    sim, a = env.sim, env.a
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if max(abs(dx), abs(dy)) == 2:
                sim.set_tile(a.x + dx, a.y + dy, block="door" if (dx, dy) == (0, 2) else "wood_wall")
            elif (dx, dy) != (0, 0):
                sim.set_tile(a.x + dx, a.y + dy, block="none")
    sim.set_tile(a.x + 1, a.y + 1, block="bed")
    env.ledger.placed.add("wood_wall")                       # (as if it had placed them)
    env.step(0)
    assert {"room_built", "room_with_door", "bed_in_room"} <= env.ledger.done
    assert "roofed_room" not in env.ledger.done


def test_attack_picks_a_slot():
    from nexel_agent.rl.actions import to_command
    obs = {"tick": 0, "self": {"inventory": {}, "z": 0},
           "entities": [{"id": 7, "kind": "agent", "dx": 1, "dy": 0}, {"id": 9, "kind": "agent", "dx": 2, "dy": 0}]}
    assert to_command(idx("attack"), obs, "S")[1]["target"] == 7
    assert to_command(idx("attack_1"), obs, "S")[1]["target"] == 9
