"""The RL env's ring verb (rl/ring.py): walls round the player from the pack, a room the world then sees."""

from nexel_agent.rl.core import SimEnv
from rl_kit import idx, neighbour_beside


def flat_round(env, r=3):
    sim, a = env.sim, env.a
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            sim.set_tile(a.x + dx, a.y + dy, ground="grass", block="none", level=sim.world.level(a.x, a.y))
    env.obs = sim.observe(a)


def test_ring_closes_a_room_with_a_door():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    env.reset(5)
    flat_round(env)
    env.a.inventory.update({"stone_wall": 7, "door": 1})
    env.obs = env.sim.observe(env.a)
    assert env.encoded()["mask"][idx("ring")]
    for _ in range(12):
        env.step(idx("ring"))
        if "room_with_door" in env.ledger.done:
            break
    assert {"room_built", "room_with_door"} <= env.ledger.done
    assert env.a.inventory.get("stone_wall", 0) == 0 and not env.a.inventory.get("door")


def test_ring_needs_the_pieces_and_nobody_beside():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, shadow_pct=0)
    env.reset(5)
    flat_round(env)
    env.a.inventory.update({"stone_wall": 5})
    env.obs = env.sim.observe(env.a)
    assert not env.encoded()["mask"][idx("ring")]                  # (5 walls do not close 8 tiles)
    env.a.inventory.update({"stone_wall": 8})
    env.obs = env.sim.observe(env.a)
    for f in env.folk:
        f.a.x, f.a.y = env.a.x + 30, env.a.y + 30
    env.obs = env.sim.observe(env.a)
    assert env.encoded()["mask"][idx("ring")]
    neighbour_beside(env)                                          # (someone beside the ring: it would wall them in)
    assert not env.encoded()["mask"][idx("ring")]
