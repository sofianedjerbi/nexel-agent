"""Helpers shared by the RL env tests."""

from nexel_agent.rl.encode import N_ACTIONS, action_name


def idx(name):
    return next(i for i in range(N_ACTIONS) if action_name(i) == name)


def during_next_tick(env, fn):
    """Something happens in the world during the next tick (as a pick-up or a blow would)."""
    step = env.sim.step

    def once():
        env.sim.step = step
        step()
        fn()
    env.sim.step = once


def neighbour_beside(env):
    sim, a, f = env.sim, env.a, env.folk[0]
    f.next = sim.tick + 1
    env.step(0)
    f.a.x, f.a.y, f.a.z = a.x + 1, a.y, 99
    f.act = 10 ** 9
    env.obs = sim.observe(a)
    return f


def to_command_for(env, name):
    from nexel_agent.rl.actions import to_command
    return to_command(idx(name), env.obs, env.facing, env.social, env.goals.goal)
