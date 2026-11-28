"""A home finished, then the world grows into it: a tree come back inside, by its door (live trial 11,
rl_runs/score/20261128-140428: felled clearing the site, it grew back in the 3x3 room by the east door; every walk to
the tile beside it, behind it, found no way, 50 minutes). It is dug out from the doorway, as a player does it."""

from test_rl_project import Project, plan_of, laid_by, BUILD, run

HUT11 = {"shape": "rect", "width": 3, "depth": 3, "storeys": 1, "door": "east",
         "roof": {"style": "flat", "material": "dirt", "eaves": "dirt", "ridge": "dirt"},
         "furniture": {"bed": [[1, 1]], "campfire": [1, 2], "chest": [2, 1], "table": [2, 2], "workbench": [0, 1]},
         "path": 2}
PACK = {"wood": 60, "plank": 30, "dirt": 60}


def test_a_tree_grown_back_inside_by_the_door_is_dug_from_the_doorway():
    env, left, n = run(HUT11, 9, 3000, PACK)
    assert not left, (n, left[:4])
    hx, hy = env.social.home
    door = next((hx + dx, hy + dy) for dx, dy, dz, p in plan_of(env.social).cells if p == "door")
    x, y = door[0] - 1, door[1]                              # (the tile inside, by the door)
    env.sim.set_tile(x, y, block="tree")
    env.a.x, env.a.y = door[0] + 3, door[1]                  # (outside, come back to it)
    env.obs = env.sim.observe(env.a)
    p, seen = Project(), []
    for _ in range(300):
        if env.sim.world.block(x, y) == "none":
            break
        a = p.act(env.obs, env.social, bool(env.encoded()["mask"][BUILD]))
        if a is None:
            break
        env.goals.goal = a[1]
        env.step(a[0])
        seen.append((env.a.x, env.a.y, a[0]))
    assert env.sim.world.block(x, y) == "none", (env.sim.world.block(x, y), seen[-6:])
    plan = plan_of(env.social)
    assert not [c for c in plan.cells if not laid_by(env.obs, env.social, plan)(c)]


def test_a_door_stood_on_is_read_as_laid_from_what_it_stands_on():
    """Live trial 13 (rl_runs/score/20261128-163110): the compound's door laid, then stood on - the grid shows "@" there,
    the door was read as not laid and a step off asked for 1907 times, the stand a tree. What it stands on, as the game
    says it ("grass+door"), is read."""
    from nexel_agent.rl.build import next_step
    from nexel_agent.rl.actions import GROUND_CH, STEP, WATER, ITEMS, tool_for
    env, left, n = run(HUT11, 9, 3000, PACK)
    assert not left, (n, left[:4])
    hx, hy = env.social.home
    door = next((hx + dx, hy + dy) for dx, dy, dz, p in plan_of(env.social).cells if p == "door")
    env.social.seen_laid = set()                             # (never seen laid: stood on since it was laid)
    env.a.x, env.a.y = door
    env.obs = env.sim.observe(env.a)
    assert "door" in env.obs["self"]["standing_on"]
    plan = plan_of(env.social)
    assert not [c for c in plan.cells if not laid_by(env.obs, env.social, plan)(c)]
    assert next_step(env.obs, env.social, ground_ch=GROUND_CH - {WATER, ITEMS}, tool_for=tool_for, dirs=STEP) is None
