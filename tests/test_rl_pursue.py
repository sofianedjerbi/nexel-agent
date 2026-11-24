"""The `pursue` verb (rl/pursue.py): picked again and again in the game's own sim, it carries out the goal - a recipe
chain from the pack and the land, a walk to a person and the ask - with no rule of when: only how."""

from nexel_agent.rl.core import SimEnv
from nexel_agent.rl.goals import Goal
from rl_kit import idx

PURSUE = idx("pursue")


def env_with(goal_kind, item=None, inv=None, person=None):
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    env.folk = []                                         # (no neighbours: the goal alone)
    env.a.inventory.clear()
    env.a.inventory.update(inv or {})
    env.goals.goal = Goal(goal_kind, item, person, env.sim.tick, env.a.count(item) if item else 0, until=10 ** 12)
    env.obs = env.sim.observe(env.a)
    return env


def run(env, n=200, done=lambda e: False):
    for _ in range(n):
        o = env.encoded()
        if not o["mask"][PURSUE] or done(env):
            break
        env.step(PURSUE)
    return env


def test_a_tool_from_raw_materials_by_pursuing_alone():
    """Wood, stone and fiber in the pack, no bench anywhere: pursue makes the bench from wood, puts it down, then the
    pickaxe (tiles.RECIPES: a pickaxe is made at a workbench)."""
    env = run(env_with("craft", "pickaxe", {"wood": 6, "stone": 3, "fiber": 1}), done=lambda e: e.a.count("pickaxe"))
    assert env.a.count("pickaxe") == 1


def test_gather_walks_to_what_drops_it_and_digs():
    env = env_with("gather", "wood")
    assert not env.encoded()["mask"][PURSUE]              # (nothing in view drops wood: finding a tree is the policy's)
    for dx in (4, 5):
        env.sim.world.set(env.a.x + dx, env.a.y - 3, block="tree")
    env.obs = env.sim.observe(env.a)
    assert env.encoded()["mask"][PURSUE]
    run(env, 300, done=lambda e: e.a.count("wood") >= 3)
    assert env.a.count("wood") >= 3


def test_no_step_no_pursue():
    env = env_with("craft", "no_such_thing")
    assert not env.encoded()["mask"][PURSUE]
    env = env_with("free")
    assert not env.encoded()["mask"][PURSUE]


def test_ask_walks_up_to_the_person_and_asks():
    """A neighbour 10 tiles off carries wood: pursue walks over, asks once in earshot; she says yes and brings it."""
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a, f = env.sim, env.a, env.folk[0]
    f.next = sim.tick + 1
    env.step(0)
    f.a.x, f.a.y, f.a.z = a.x + 10, a.y, 99
    sim.height(f.a)
    f.until, f.act = sim.tick + 10 ** 6, sim.tick + 10 ** 6       # (she stands there till asked)
    f.rng.random = lambda: 0.0                                     # (she says yes)
    f.a.inventory["wood"] = 2
    env.goals.goal = Goal("ask", "wood", f.a.name, sim.tick, until=10 ** 12)
    env.obs = sim.observe(a)
    for _ in range(80):
        if a.count("wood"):
            break
        if env.encoded()["mask"][PURSUE]:
            env.step(PURSUE)
        else:
            f.act = sim.tick                                       # (asked: now she acts, and brings it)
            env.step(0)
    assert a.count("wood") >= 1 and env.goals.counts["ask"][0] == 1
