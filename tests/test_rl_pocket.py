"""Shut in a pocket of open ground (live trial 12, rl_runs/score/20261128-155737: a hut's stone walls begun before the
design came, the garden's fence laid across their open side, the body inside; 50 minutes of walks to stands
outside, "no path", the pieces tried from where it stood, "out of range"). Its way out is dug through what is not
the home's, walked to inside the pocket, as a player breaks out of a pen."""

from test_rl_project import HOUSE, run


def test_shut_in_a_pocket_of_walls_not_the_homes_it_digs_out_and_builds():
    from nexel_agent.rl.core import SimEnv
    reset = SimEnv.reset

    def walled(self, seed):
        out = reset(self, seed)
        x, y = self.a.x, self.a.y
        for i in range(-2, 4):                               # (a pocket three tiles long, walls of stone all round)
            for j in (-1, 1):
                self.sim.set_tile(x + i, y + j, block="stone_wall")
        for i in (-2, 3):
            self.sim.set_tile(x + i, y, block="stone_wall")
        for i in range(-1, 3):
            self.sim.set_tile(x + i, y, block="none")
        self.obs = self.sim.observe(self.a)
        return out

    SimEnv.reset = walled
    try:
        env, left, n = run(HOUSE, 9, 3000)
    finally:
        SimEnv.reset = reset
    assert not left, (n, left[:4], (env.a.x, env.a.y))
