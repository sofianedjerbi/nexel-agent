"""A build is a project the body works at to its end (#87): from an empty pack, the design drawn stands through the
project alone - the material gathered from the land as the game's recipes say, made, laid."""

from types import SimpleNamespace

import pytest

from nexel_agent.rl.build import laid_by, plan_of
from nexel_agent.rl.core import SimEnv
from nexel_agent.rl.dug import design_of
from nexel_agent.rl.plan import of_structure
from nexel_agent.rl.project import BUILD, Project, lacking

HOUSE = {"width": 3, "depth": 3, "door": "south", "roof": {"style": "gable"},
         "furniture": {"bed": [0, 0], "campfire": [2, 0], "chest": [0, 2], "chair": [2, 2]}}


def run(spec, seed, steps, pack=None):
    env = SimEnv(ep_ticks=10 ** 7, roam=0, rivals=0)
    env.reset(seed)
    env.folk = []
    env.goals = SimpleNamespace(goal=None, step=lambda *a: 0, draw=lambda *a, **k: None, counts={})
    env.social.plan = of_structure(design_of(spec))
    env.a.inventory.clear()
    env.a.inventory.update(pack or {})
    p, n = Project(), 0
    for n in range(steps):
        a = p.act(env.obs, env.social, bool(env.encoded()["mask"][BUILD]))
        if a is None:
            break
        env.goals.goal = a[1]
        env.step(a[0])
    plan = plan_of(env.social)
    left = [c for c in plan.cells if not laid_by(env.obs, env.social, plan)(c)] if env.social.home else plan.cells
    return env, left, n


def test_the_bill_is_the_least_raw_material_a_demand_takes():
    """Research notes, Theorem 1. A recipe yielding two (a wood: 2 shingles) is crafted ceil(n / 2) times for n,
    not once a cell (107 shingles: 54 wood, not 107); a station once; what the pack holds first."""
    from collections import Counter

    from nexel_agent.rl.project import bill
    assert bill(Counter({"roof": 107}), {}) == {"wood": 54}
    assert bill(Counter({"wood_wall": 4}), {}) == {"wood": 6 + 4}          # (2 crafts of 3 wood, a workbench of 4)
    assert bill(Counter({"wood_wall": 4}), {"workbench": 1, "wood": 2}) == {"wood": 4}
    assert bill(Counter({"roof": 3, "plank": 3}), {"roof": 1}) == {"wood": 1 + 2}


def test_what_a_plan_lacks_is_read_down_to_the_land():
    plan = of_structure(design_of(HOUSE))
    need = lacking(plan, lambda c: False, {})
    assert set(need) <= {"wood", "stone", "fiber"} and need["wood"] > 50
    assert lacking(plan, lambda c: True, {}) == {}


@pytest.mark.parametrize("seed", [7])
def test_from_an_empty_pack_a_house_stands_through_the_project_alone(seed):
    env, left, n = run(HOUSE, seed, 6000)
    assert env.social.home is not None and left == [], (n, len(left), left[:6], dict(env.a.inventory))


def test_the_build_the_mind_names_is_kept_whatever_its_goal_till_another_or_none(tmp_path):
    """Live trial 6: the mind named "a 2-storey round house with gabled shingle roof" with a gather goal, twice, and
    both were dropped (the words were kept only with the goal build)."""
    from nexel_agent.rl.mind import Mind
    m = Mind("Rook", "", "http://x", None, tmp_path / "m.json")
    m.apply({"goal": "gather", "item": "fiber", "build": "a round house with a gabled shingle roof"})
    assert m.build_words == "a round house with a gabled shingle roof" and m.goal.kind == "gather"
    m.apply({"goal": "craft", "item": "axe", "build": ""})
    assert m.build_words == "a round house with a gabled shingle roof"
    m.apply({"goal": "gather", "item": "wood", "build": "none"})
    assert m.build_words is None


def test_a_flight_of_stairs_is_counted_a_stairs_item_a_step():
    from test_rl_structure import DESIGNS
    s = design_of(DESIGNS["tower"])
    plan = of_structure(s)
    steps = sum(len(st) for _, st, _, _ in s.stairs)
    only_stairs = lambda c: c[3] != "stairs"                                 # noqa: E731  (all else laid)
    assert lacking(plan, only_stairs, {}) == {"wood": -(-steps // 2)}       # (a wood makes 2 stairs)




def test_a_plank_laid_up_a_storey_is_not_read_as_the_wall_standing_on_it():
    """The tower (#91): kept by its place alone, the first floor's plank read as the wall on it laid; the wall never
    went up, the second flight ran to the first floor and the top storey was never reached."""
    from nexel_agent.rl.social import SocialMemory
    s = SocialMemory("Rook", ("wood",))
    s.achieved("place", {"item": "plank", "x": 4, "y": 2, "z": 3}, True, 0, 99)
    assert (4, 2, 3, "block") in s.up_laid and (4, 2, 3, "thing") not in s.up_laid
    s.achieved("place", {"item": "wood_wall", "x": 4, "y": 2, "z": 3}, True, 0, 99)
    assert (4, 2, 3, "thing") in s.up_laid


def test_a_flight_waits_for_its_doors_column():
    from test_rl_structure import DESIGNS
    plan = of_structure(design_of(DESIGNS["tower"]))
    top = [c for c in plan.cells if c[3] == "stairs"][-1]
    after = plan.after[top]
    assert any(c[3] == "plank" and c[2] == 6 for c in after) and any(c[3] == "wall" and c[2] == 3 for c in after)
    assert not any(c[3] == "door" for c in after)       # (Theorem 2: a door or nothing; waiting on it, seed 107 deadlocked)
    assert not plan.held(top, lambda c: c not in after) and plan.held(top, lambda c: True)


@pytest.mark.parametrize("seed", [9])
def test_from_an_empty_pack_a_tower_of_three_storeys_stands(seed):
    from test_rl_structure import DESIGNS
    env, left, n = run(DESIGNS["tower"], seed, 2000)
    assert left == [], (n, len(left), left[:4])


def test_unhurt_and_unsteered_the_body_moves_eats_and_talks_but_makes_nothing_of_its_own():
    """Trial 7: with the build waiting, the learned body placed 22 workbenches; trial 5: it dug its own site."""
    from nexel_agent.rl.actions import ACTIONS
    from nexel_agent.rl.project import CALM
    kinds = {a[0] for a, ok in zip(ACTIONS, CALM, strict=True) if ok}
    assert not kinds & {"craft", "place", "dig", "dig_at", "dig_down"}
    assert {"step", "attack", "eat", "pursue", "build", "say"} <= kinds


def test_a_thought_that_fails_is_kept_with_its_cause(tmp_path):
    """Trial 7: 75 thoughts failed and nothing said why."""
    import asyncio

    from nexel_agent.rl.mind import Mind
    m = Mind("Rook", "", "http://x", None, tmp_path / "m.json")
    asyncio.run(m.think({"self": {}}))
    log = (tmp_path / "mind_errors.log").read_text()
    assert m.stats["errors"] == 1 and "Error" in log and not m.busy


def test_inside_its_own_site_the_builder_steps_off_to_put_a_bench_down():
    """The round home, seed 9: standing inside the site, every tile round it the home's, the bench had nowhere to
    go; the walls were skipped and 80 stayed unlaid with the wood in hand."""
    from test_rl_structure import DESIGNS
    env, left, n = run(DESIGNS["round home"], 9, 1500)
    assert left == [], (n, len(left))


def test_a_clearing_dig_refused_from_one_side_is_tried_from_another():
    """Seed 21: a mushroom on the tower's site "behind the land" from where the builder stood, dug at 199 times."""
    from nexel_agent.rl.site import TRIES, tried
    s = SimpleNamespace()
    assert not any(tried(s, 3, 4, (3, 3)) for _ in range(TRIES))
    assert tried(s, 3, 4, (3, 3)) and not tried(s, 3, 4, (2, 4), count=False)


def test_from_an_empty_pack_the_tower_stands_where_its_site_had_land_in_the_way():
    from test_rl_structure import DESIGNS
    env, left, n = run(DESIGNS["tower"], 21, 6000)
    assert left == [], (n, len(left))


def test_no_site_has_land_under_the_eaves_closer_than_a_storey():
    """The round home, seed 21: a hill one layer up under its eaves; the game refuses a block "too low over the floor:
    leave room to stand under it", 199 times. The site check now leaves a storey under every overhanging cell."""
    from test_rl_structure import DESIGNS
    env, left, n = run(DESIGNS["round home"], 21, 3000)
    assert left == [], (n, len(left))


def test_walled_in_by_its_own_wall_rocks_and_a_bush_the_builder_digs_its_way_out():
    """The T on seed 7: its own wall east, rocks south and west, a berry bush north; every walk "no path" 199 times."""
    from test_rl_wings import WINGS
    spec = {**WINGS["T"], "roof": {"style": "gable", "material": "roof", "ridge": "plank"}}
    env, left, n = run(spec, 7, 2000)
    assert left == [], (n, len(left))


def test_a_bench_is_not_put_down_on_a_hills_rim():
    """The cross on seed 109: the game keeps nothing on a hill's edge ("cliff_edge"); the bench went there 199 times."""
    from test_rl_wings import WINGS
    spec = {**WINGS["cross"], "roof": {"style": "hip", "material": "roof", "eaves": "plank"}}
    env, left, n = run(spec, 109, 2000)
    assert left == [], (n, len(left))


def test_where_a_flight_is_laid_from_is_part_of_the_site():
    """The tower on seed 107: a tree and the builder's own bench on the two tiles its first flight is laid from."""
    from test_rl_structure import DESIGNS
    s = design_of(DESIGNS["tower"])
    assert all(p in s.clear() for _, _, at, _ in s.stairs for p in at)
    env, left, n = run(DESIGNS["tower"], 107, 2000)
    assert left == [], (n, len(left))


TRIAL8 = {"shape": "rect", "width": 4, "depth": 3, "storeys": 2, "door": "east",
          "roof": {"style": "gable", "material": "roof", "eaves": "plank", "ridge": "plank"},
          "furniture": {"bed": [[1, 1]], "campfire": [1, 2], "chest": [2, 2], "table": [2, 1], "chair": [3, 1]}, "path": 2}


def test_a_gables_ridge_stops_flush_with_its_end_walls_so_every_cell_is_in_reach():
    """Live trial 8: a two-storey gabled house stood at 201 of 203; the two left were the ridge's ends, 9 layers up
    over the overhang past the end walls, out of reach from every stand. The ridge now stops at the walls."""
    s = design_of(TRIAL8)
    foot = s.inside | s.ring
    top = max(c[2] for c in s.cells())
    assert all(c[:2] in foot for c in s.cells() if c[2] == top)
    env, left, n = run(TRIAL8, 7, 2000)
    assert left == [], (n, len(left), left[:3])


def test_a_station_is_never_put_on_its_last_way_out():
    """The house on seed 107: between a tree, its own wall and a tree, the workbench put on the one open side, then
    dug back up to get out (way_out), 611 times; 120 cells left at 7999 decisions. A station goes down with a way
    still open, else a step off first."""
    env, left, n = run(HOUSE, 107, 1500)
    assert not left, (n, left[:4])


def test_a_stand_with_no_way_to_it_is_given_up_for_another():
    """The T's gabled roof on seed 100: a ring's walk to its stand had no path, ordered 2585 times, 80 roof cells
    left. A ring's walk counts as a walk: shut, the stand is left for another."""
    from test_rl_wings import WINGS
    env, left, n = run({**WINGS["T"], "roof": {"style": "gable", "material": "roof", "ridge": "plank"}}, 100, 2000)
    assert not left, (n, left[:4])


def test_a_site_found_uneven_is_looked_past_not_given_up():
    """The tower on seed 202: the site found was not flat, dropped, and no other in view - the builder said nothing to
    do and the executor stopped at its first decision. It looks further instead."""
    from test_rl_structure import DESIGNS
    env, left, n = run(DESIGNS["tower"], 202, 1500)
    assert not left, (n, left[:4])


def test_a_site_whose_ground_is_out_of_view_is_kept():
    """The T on seed 203: gathering beside its site, the site's first ground tile out of view read as uneven, and the
    site (nothing laid yet) given up, four times, then the search ran out. Its level is read on what is in view."""
    from test_rl_wings import WINGS
    env, left, n = run({**WINGS["T"], "roof": {"style": "gable", "material": "roof", "ridge": "plank"}}, 203, 1500)
    assert not left, (n, left[:4])


def test_a_walk_over_the_land_ends_on_open_ground():
    """The house on seed 209: a search's walks, and the walk back home, each ended in a river (no path, 2281 times),
    and nothing was laid. A walk over the land ends on the furthest open ground on its line."""
    env, left, n = run(HOUSE, 209, 1500)
    assert not left, (n, left[:4])


def test_a_house_stops_short_of_a_drop_and_turns_off_a_shut_way_home():
    """Seeds 306 and 307: looking for fibre, walks went down cliffs into a pit with no way back up, and the walk home,
    shut, was ordered 1352 times. A land walk ends short of a drop of more than a step, and a shut way home turns to
    the open way nearest it."""
    from test_rl_structure import DESIGNS
    for spec, seed in ((HOUSE, 306), (DESIGNS["two storeys"], 307)):
        env, left, n = run(spec, seed, 1500)
        assert not left, (seed, n, left[:4])


def test_shut_in_by_its_own_walls_and_trees_it_clears_its_way_out():
    """The round home on seed 203: its own walls south and west, trees on its site north and east; way_out dug only
    what is not the home's, and the door's stand was walked to 713 times. A tree on its own ground is dug out, a shut
    stand never chosen."""
    from test_rl_structure import DESIGNS
    env, left, n = run(DESIGNS["round home"], 203, 1500)
    assert not left, (n, left[:4])


def test_a_way_shut_before_a_storey_stood_is_tried_again():
    """The tower on seed 304: its last roof cell's stand, up on the top floor, was walked to before the flight there
    was laid, shut for good, and the build ended one cell short. Shut ways are tried again once the build grows."""
    from test_rl_structure import DESIGNS
    env, left, n = run(DESIGNS["tower"], 304, 1500)
    assert not left, (n, left[:4])


def test_shut_in_a_notch_of_its_walls_it_digs_out():
    """The T on seed 504: in a notch of its own walls, by two trees under its roof's overhang (its own tiles, never
    cleared), every way off shut (357 walks), no workbench put down, the build ended with 254 cells left. Shut in, it
    digs through a tree, even on its own ground."""
    from test_rl_wings import WINGS
    env, left, n = run({**WINGS["T"], "roof": {"style": "gable", "material": "roof", "ridge": "plank"}}, 504, 3000)
    assert not left, (n, left[:4])


ROOF10 = {"style": "flat", "material": "stone_block", "eaves": "plank", "ridge": "plank"}
LIVE10 = {"parts": [                                         # (live trial 10's draft, rl_runs/score/20261128-114221)
    {"width": 5, "depth": 4, "storeys": 2, "door": "south", "roof": ROOF10, "path": 2,
     "furniture": {"bed": [[2, 1]], "campfire": [2, 2], "chest": [4, 1], "table": [1, 1, 0], "chair": [1, 2, 0],
                   "workbench": [3, 1, 0]}},
    {"at": "east", "width": 5, "depth": 4, "storeys": 2, "door": "south", "roof": ROOF10, "path": 3,
     "furniture": {"bed": [[2, 2]], "campfire": [[2, 3]], "chest": [[1, 1]], "table": [[4, 1]], "chair": [[4, 2]]}}]}


@pytest.mark.parametrize("seed", [9, 21])
def test_a_stone_roofed_house_and_its_annex_stand_from_a_full_pack(seed):
    """Live trial 10's design: two storeys under a flat roof of stone on a plank deck, an annex of the same beside it.
    A stone block laid is land, and the view's heights show the roof's top over its rooms: read as the land, the site
    was 'dug into' from the annex's end, a pit walked to and back 45 minutes live (5,521 decisions on seed 9); on 21 the
    annex ran into water unseen when the site was chosen. The heights read on the build itself are not the land's;
    water on the site is filled with earth from the pack."""
    env, left, n = run(LIVE10, seed, 1500, pack={"stone_block": 400, "wood": 300})
    assert not left, (n, left[:4])
