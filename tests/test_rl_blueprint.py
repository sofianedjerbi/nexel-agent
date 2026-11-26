"""The architect's drawings (#66): a tile grid read, stamped and judged before a block is laid (rl/blueprint.py), and
the draw -> judge -> redraw loop on a scripted model (rl/architect.py)."""

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


def test_a_drawn_home_is_laid_by_building_alone_and_judged_as_drawn():
    """The build verb lays a drawing as it lays the hut: from the pack, standing on its floor, the roof last."""
    from rl_kit import idx

    from nexel_agent.rl.build import drawn
    from nexel_agent.rl.core import SimEnv
    big = "#######\n#b.f.c#\n#.....#\n#h....#\n###D###\n   =\n   =\n"
    rows = blueprint.parse(big)
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    env.folk = []
    env.social.plan = drawn(rows)
    env.a.inventory.clear()
    env.a.inventory.update({"stone_wall": 30, "door": 1, "bed": 1, "campfire": 1, "chest": 1, "chair": 1, "path": 2,
                            "roof": 63})
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
