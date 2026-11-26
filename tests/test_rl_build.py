"""The RL env's ring verb (rl/ring.py): walls round the player from the pack, a room the world then sees."""

from nexel_agent.rl.core import SimEnv
from rl_kit import idx, neighbour_beside


def flat_round(env, r=3):
    sim, a = env.sim, env.a
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            sim.set_tile(a.x + dx, a.y + dy, ground="grass", block="none", level=sim.world.level(a.x, a.y))
    env.obs = sim.observe(a)


def ring_ok(env):
    """ring's own rules (rl/ring.py plan): the policy no longer picks it (retired in the mask: build lays rooms)."""
    from nexel_agent.rl import ring
    from nexel_agent.rl.actions import GROUND_CH
    assert not env.encoded()["mask"][idx("ring")]
    return ring.plan(env.obs, GROUND_CH) is not None


def test_ring_closes_a_room_with_a_door():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    env.reset(5)
    flat_round(env)
    env.a.inventory.update({"stone_wall": 7, "door": 1})
    env.obs = env.sim.observe(env.a)
    assert ring_ok(env)
    for _ in range(12):
        env.step(idx("ring"))
        if "room_with_door" in env.ledger.done:
            break
    assert {"room_built", "room_with_door"} <= env.ledger.done
    assert env.a.inventory.get("stone_wall", 0) == 0 and not env.a.inventory.get("door")


def paid(env, name):
    return {n for k, n in env.marks.paid if k == name}


def ring_until(env, done, tries=12):
    for _ in range(tries):
        env.step(idx("ring"))
        if done():
            break


def test_fill_and_stock_steps_paid_once_for_its_own_pieces():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    env.reset(5)
    flat_round(env)
    sim, a = env.sim, env.a
    for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        sim.set_tile(a.x + dx, a.y + dy, block="stone_wall")       # (walls it did not place: no fill step)
    a.inventory.update({"stone_wall": 8})
    env.obs = sim.observe(a)
    env.step(0)
    assert not paid(env, "fill") and paid(env, "stock") == {4, 8}  # (8 held at once)
    ring_until(env, lambda: a.inventory.get("stone_wall", 0) <= 4)
    assert paid(env, "fill") == {4}                                # (4 of its own round it; the corners are not its)
    assert env.ledger.counts["multi_place"] == 1                   # (2+ of its own in one decision)
    for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        sim.set_tile(a.x + dx, a.y + dy, block="none")
    env.obs = sim.observe(a)
    ring_until(env, lambda: 8 in paid(env, "fill"))
    assert paid(env, "fill") == {4, 6, 8}
    for d in ((0, -1), (1, 0)):                                    # (broken and put back: not paid again)
        sim.set_tile(a.x + d[0], a.y + d[1], block="none")
    a.inventory.update({"stone_wall": 2})
    env.obs = sim.observe(a)
    before = set(env.marks.paid)
    _, r, *_ = env.step(idx("ring"))
    env.step(0)
    assert env.marks.paid == before and r < 0.1


def test_ring_places_what_it_holds_and_never_beside_someone():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, shadow_pct=0)
    env.reset(5)
    flat_round(env)
    for f in env.folk:
        f.a.x, f.a.y = env.a.x + 30, env.a.y + 30
    env.a.inventory.update({"stone_wall": 1})
    env.obs = env.sim.observe(env.a)
    assert not ring_ok(env)                  # (one piece: place does that)
    env.a.inventory.update({"stone_wall": 3})
    env.obs = env.sim.observe(env.a)
    assert ring_ok(env)
    neighbour_beside(env)                                          # (someone beside the ring: it would wall them in)
    assert not ring_ok(env)
