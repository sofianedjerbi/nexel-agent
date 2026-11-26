"""The architect's drawings (#66): a tile grid read, stamped and judged before a block is laid (rl/blueprint.py), and
the draw -> judge -> redraw loop on a scripted model (rl/architect.py)."""

import json

import pytest

from nexel_agent.rl import architect, blueprint, judge

GOOD = """
#####
#b.f#
#...#
#c.h#
##D##
  =
  =
"""


def test_a_good_home_drawn_passes_every_check_it_can():
    j = blueprint.judged(GOOD)
    assert j["ok"] and j["failed"] == [] and j["grade"] == j["of"], j
    assert j["of"] == len(judge.CHECKS) + len(judge.SENSE) + len(judge.LOOKS) - len(blueprint.CANNOT)


@pytest.mark.parametrize("change, failed", [
    ({(2, 4): "#"}, "v1.door"),                       # no door: walled in
    ({(4, 2): " "}, "v1.closed"),                     # a gap in the east wall
    ({(1, 1): "."}, "v1.bed"),                        # no bed
    ({(1, 3): "."}, "v2.chest"),                      # no chest
    ({(2, 3): "t"}, "v2.reachable"),                  # a table in front of the door: nothing reached past it
    ({(2, 5): " ", (2, 6): " "}, "v3.path"),          # no path out
])
def test_each_fault_drawn_is_named(change, failed):
    rows = [list(r) for r in GOOD.strip("\n").splitlines()]
    for (x, y), ch in change.items():
        rows[y][x] = ch
    j = blueprint.judged("\n".join("".join(r) for r in rows))
    assert failed in j["failed"], j["failed"]
    assert j["told"] != "nothing failed"


def test_what_is_no_drawing_is_told_not_judged():
    for text, why in (("", "no grid"), ("#?#", "unknown"), ("...", "no walls"), ("#" * 20, "too big")):
        j = blueprint.judged(text)
        assert not j["ok"] and why in j["error"]


def test_the_pieces_go_down_shell_first_and_the_roof_last_over_the_walls_and_past():
    ps = blueprint.pieces(blueprint.parse(GOOD))
    kinds = [p for _, _, p in ps]
    assert kinds.index("bed") > max(i for i, k in enumerate(kinds) if k in ("wall", "door"))
    assert kinds[-1] == "roof" and kinds.index("roof") > kinds.index("path")
    roof = {(x, y) for x, y, p in ps if p == "roof"}
    assert (-1, -1) in roof and (5, 5) in roof and len(roof) == 7 * 7


def test_the_architect_draws_again_with_what_failed_and_keeps_the_best():
    replies = iter(["```\n#####\n#...#\n#...#\n#####\n```", f"```{GOOD}```"])
    seen = []

    def ask(msgs):
        seen.append(msgs[-1]["content"])
        return next(replies)
    d = architect.design("a home for one", ask)
    assert d["history"][0] < d["history"][1] == d["of"] and d["failed"] == []
    assert "a home for one" in seen[0] and "no door" in seen[1]          # (the second round was told what failed)


def test_a_reply_without_a_fence_is_read_whole():
    assert architect.grid_of(GOOD) == GOOD and architect.grid_of(f"here:\n```text\n{GOOD.strip()}\n```") == GOOD.strip() + "\n"


CROWDED = "#####\n#.b.#\n#c.t#\n#f.h#\n##D##\n  =\n  =\n"    # (bench run 7: laid from its corner, it walled itself in)


@pytest.mark.parametrize("grid", ["#######\n#b.f.c#\n#.....#\n#h....#\n###D###\n   =\n   =\n", CROWDED])
def test_a_drawn_home_is_laid_by_building_alone_and_judged_as_drawn(grid):
    """The build verb lays a drawing as it lays the hut: from the pack, standing on the floor the door leads to (never
    where the furniture closes round it) or round the walls outside, the roof last."""
    from collections import Counter

    from rl_kit import idx

    from nexel_agent.rl.build import drawn
    from nexel_agent.rl.core import SimEnv
    rows = blueprint.parse(grid)
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    env.folk = []
    env.social.plan = drawn(rows)
    env.a.inventory.clear()
    env.a.inventory.update(Counter("stone_wall" if p == "wall" else p for _, _, p in blueprint.pieces(rows)))
    env.obs = env.sim.observe(env.a)
    for _ in range(400):
        if not env.encoded()["mask"][idx("build")]:
            break
        env.step(idx("build"))
    if "roofs" not in env.sim.observe(env.a):
        pytest.skip("the game's observation shows no roofs yet (sofianedjerbi/nexel-worlds#45)")
    hx, hy = env.social.home
    px, py = blueprint.inside_point(rows)
    w = env.sim.world
    assert judge.verdict(w, hx + px, hy + py)["failed"] == [], judge.told(judge.verdict(w, hx + px, hy + py))
    assert judge.sense(w, hx + px, hy + py)["failed"] == []
    assert [c for c in judge.looks(w, hx + px, hy + py)["failed"] if f"v3.{c}" not in blueprint.CANNOT] == []


def test_live_a_home_not_begun_waits_for_its_drawing_and_keeps_it(tmp_path):
    import time
    from types import SimpleNamespace

    from nexel_agent.rl.build import HUT
    from nexel_agent.rl.core import SimEnv
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    obs = env.sim.observe(env.a)
    social, stats = SimpleNamespace(home=None, plan=None), {}
    d = architect.Drafting(lambda msgs: f"```{GOOD}```", tmp_path / "plan.json", "a home for one")
    assert not d.holds(social, False, obs)                                # (not building: no drawing)
    while d.holds(social, True, obs, stats):
        time.sleep(0.01)
    assert social.plan.pieces == blueprint.pieces(blueprint.parse(GOOD)) and stats["draft"]["grade"] == stats["draft"]["of"]
    social2 = SimpleNamespace(home=(3, 4), plan=None)                    # (the next session: the drawing handed back)
    assert not architect.Drafting(None, tmp_path / "plan.json", "").holds(social2, True, obs)
    assert social2.plan.pieces == social.plan.pieces and social2.plan is not HUT


def test_live_a_drawing_short_of_full_marks_leaves_the_hut(tmp_path):
    import time
    from types import SimpleNamespace

    from nexel_agent.rl.core import SimEnv
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    social = SimpleNamespace(home=None, plan=None)
    d = architect.Drafting(lambda msgs: "```\n#####\n#...#\n#####\n```", tmp_path / "plan.json", "")
    while d.holds(social, True, env.sim.observe(env.a)):
        time.sleep(0.01)
    assert social.plan is None and not (tmp_path / "plan.json").exists() and d.last["grade"] < d.last["of"]


DESIGN = {"width": 3, "depth": 3, "door": "south", "path": 2,
          "furniture": {"bed": [0, 0], "campfire": [2, 0], "chest": [0, 2], "chair": [2, 2]}}


@pytest.mark.parametrize("door", blueprint.SIDES)
def test_a_design_chosen_is_drawn_whole_whatever_its_doors_side(door):
    j = blueprint.judged(blueprint.render({**DESIGN, "door": door}))
    assert j["failed"] == [], (door, j["told"])


def test_a_design_that_cannot_be_drawn_is_told_why_and_two_of_a_piece_go_down():
    for bad, why in (({"width": 30}, "width and depth"), ({"door": "up"}, "door is one of"),
                     ({"furniture": {"bed": [5, 0]}}, "outside"), ({"furniture": {"sofa": [0, 0]}}, "no such furniture"),
                     ({"furniture": {"bed": [0, 0], "chest": [0, 0]}}, "another piece")):
        reply = json.dumps({**DESIGN, **bad})
        j = blueprint.judged(architect.grid_of(reply))
        assert not j["ok"] and why in j["error"], (bad, j)
    two = blueprint.render({**DESIGN, "width": 5, "furniture": {**DESIGN["furniture"], "bed": [[0, 0], [4, 0]]}})
    assert two.count("b") == 2


def test_what_is_out_of_reach_is_named():
    j = blueprint.judged(blueprint.render({**DESIGN, "furniture": {**DESIGN["furniture"], "table": [1, 2]}}))
    assert "v2.reachable" in j["failed"] and "not reached on foot from the door" in j["told"]
