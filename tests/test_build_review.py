"""The build review page (#33, lite: no renders): rooms found and judged near the player in the sim
(tools/sim_builds.py), drawn top-down beside the judge's reports (tools/build_review.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_review as B  # noqa: E402
import sim_builds as SB  # noqa: E402

from nexel_agent.rl.core import SimEnv  # noqa: E402


def test_a_room_near_the_player_is_found_judged_and_drawn():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a = env.sim, env.a
    x, y = a.x + 5, a.y + 5                                 # 3 by 3 inside, its door north, a bed in it
    for i in range(-2, 5):
        for j in range(-2, 5):
            sim.world.set(x + i, y + j, ground="grass", block="none")
    for i in range(-1, 4):
        for j in range(-1, 4):
            if i in (-1, 3) or j in (-1, 3):
                sim.world.set(x + i, y + j, block="door" if (i, j) == (1, -1) else "wood_wall")
    sim.world.set(x + 2, y + 2, block="bed")
    rows = SB.rooms(sim.world, [(a.x, a.y)])
    assert len(rows) == 1 and rows[0]["v1"] >= 4 and rows[0]["facts"]["area"] == 9
    m = rows[0]["map"]
    assert len(m["blocks"]) == 3 + 2 + 2 and m["inside"][2] == "0011100"      # (the room and 2 tiles round it)
    html = B.page([("r1", "Rook", [{**rows[0], "by": "Rook"}])], [{"checkpoint": "rl_runs/r60/best.zip", "seeds": [1, 2],
                                                                    "rooms": [{**rows[0], "seed": 1}]}])
    assert "Run r1: Rook's rooms (1)" in html and "In the sim: best.zip" in html and "seed 1 · at" in html
    assert html.count('<table class="g">') == 2 and "wood wall" in html and 'title="bed' in html
