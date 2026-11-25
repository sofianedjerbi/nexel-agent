"""The build judge (rl/judge.py, v1): the game's reading of a build as checks a player would name."""

from nexel_agent.rl import judge
from nexel_agent.rl.core import SimEnv


def ring(sim, x, y, w, h, door=None):
    for dx in range(-1, w + 1):
        for dy in range(-1, h + 1):
            if dx in (-1, w) or dy in (-1, h):
                sim.world.set(x + dx, y + dy, block="door" if (x + dx, y + dy) == door else "stone_wall")


def test_a_room_grows_check_by_check():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    assert judge.verdict(sim.world, a.x, a.y)["grade"] == 0
    x, y = a.x - 1, a.y - 1
    ring(sim, x, y, 3, 3)
    v = judge.verdict(sim.world, a.x, a.y)
    assert v["checks"]["closed"] and v["checks"]["room_enough"] and not v["checks"]["door"] and v["grade"] == 2
    ring(sim, x, y, 3, 3, door=(x + 1, y - 1))
    sim.world.set(x + 2, y + 2, block="bed")
    sim.world.set(x, y + 2, block="campfire")
    v = judge.verdict(sim.world, a.x, a.y)
    assert v["failed"] == ["roofed"] and v["grade"] == 5
    assert judge.told(v).startswith("closed, a door, no roof (0% covered), lit, a bed")


def test_the_build_potential_pays_as_it_grows_and_back_when_torn_down():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    p, ok = judge.Potential(), [("place", {}, {"ok": True})]
    x, y = a.x - 1, a.y - 1
    ring(sim, x, y, 3, 3)
    assert p.step(sim.world, None, (a.x, a.y), ok) == 2 * judge.Potential.UNIT        # (closed, room enough)
    assert p.step(sim.world, None, (a.x + 9, a.y), ok) == 0                              # (walked out: no loss)
    sim.world.set(x + 1, y - 1, block="none")
    assert p.step(sim.world, None, (a.x + 9, a.y), [("dig", {}, {"ok": True})]) == -2 * judge.Potential.UNIT
