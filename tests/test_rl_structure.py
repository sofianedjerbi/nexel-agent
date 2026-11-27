"""Builds in three dimensions (#66): a design of storeys, a rectangle or a round, drawn (rl/structure.py), judged
storey by storey (rl/judge3d.py) and laid by the build verb alone in the game's own sim (rl/plan.py, rl/build.py)."""

import pytest

from nexel_agent.rl import architect, judge3d
from nexel_agent.rl.structure import UPSTAIRS_FURNITURE, Structure, footprint

HOME = {"bed": [0, 0], "campfire": [2, 0], "chest": [0, 2], "chair": [2, 2]}
DESIGNS = {
    "house": {"width": 3, "depth": 3, "door": "south", "furniture": HOME},
    "two storeys": {"width": 3, "depth": 3, "door": "south", "storeys": 2,
                    "furniture": {"campfire": [2, 0], "chest": [0, 2], "bed": [0, 0], "table": [1, 0, 1], "chair": [0, 0, 1]}},
    "round base": {"shape": "round", "radius": 4, "door": "south", "roof": False, "furniture": {"campfire": [3, 3]}},
    "round home": {"shape": "round", "radius": 3, "door": "south", "furniture": HOME},
    "tower": {"width": 3, "depth": 3, "door": "south", "storeys": 3,
              "furniture": {"campfire": [2, 0], "chest": [0, 2], "bed": [0, 0], "table": [0, 0, 1], "chair": [1, 0, 2]}},
}


@pytest.mark.parametrize("name", DESIGNS)
def test_each_kind_of_build_designed_well_passes_every_check(name):
    j = judge3d.judged(DESIGNS[name])
    assert j["ok"] and j["failed"] == [], (name, j["failed"])


def test_a_round_is_walled_whole_and_a_wall_gone_opens_it():
    inside, ring = footprint({"shape": "round", "radius": 4})
    near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]   # noqa: E731
    assert all(q in inside or q in ring for p in inside for q in near(p))           # (no way out but through a wall)
    s = Structure(DESIGNS["round base"])
    s.storeys[0].pop(next(p for p, t in s.storeys[0].items() if t == "wall"))
    from nexel_agent.rl.core import SimEnv
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    lv = judge3d.stamp(env.sim, s, env.a.x - 24, env.a.y - 24)
    assert "closed" in judge3d.judge_at(env.sim, s, env.a.x - 24, env.a.y - 24, lv)["failed"]


def test_upstairs_takes_only_what_the_game_stands_on_a_floor_and_says_so():
    assert "bed" in UPSTAIRS_FURNITURE and "campfire" not in UPSTAIRS_FURNITURE
    j = judge3d.judged({**DESIGNS["two storeys"], "furniture": {"campfire": [0, 0, 1]}})
    assert not j["ok"] and "needs the land under it" in j["error"]
    for bad, why in (({"storeys": 4}, "storeys from 1 to"), ({"shape": "star"}, "shape is one of"),
                     ({"shape": "round", "radius": 9}, "radius from 2 to 6")):
        assert why in judge3d.judged({**DESIGNS["house"], **bad})["error"]


def test_the_storeys_go_up_as_the_game_holds_them():
    """A floor's planks over the walls first, then each beside one laid; a flight of stairs from the ground to each
    upper storey's door, longer for each storey up."""
    s = Structure(DESIGNS["tower"])
    cells = s.cells()
    planks = [(x, y) for x, y, z, p in cells if p == "plank" and z == 3]
    walls = {p for p, t in s.storeys[0].items() if t in ("wall", "door")}
    seen = set()
    for i, (x, y) in enumerate(planks):
        assert (x, y) in walls or any(q in seen for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))), (i, x, y)
        seen.add((x, y))
    assert [len(steps) for _, steps, _, _ in s.stairs] == [2, 5]
    assert max(z for *_, z, p in cells if p == "roof") == 9


@pytest.mark.parametrize("name, seed", [("two storeys", 21), ("round base", 7), ("tower", 9)])
def test_a_build_of_storeys_or_a_round_is_laid_whole_by_building_alone(name, seed):
    b = judge3d.built(DESIGNS[name], seed=seed)
    assert b["sited"] and b["failed"] == [], b


def test_the_architect_chooses_a_build_of_storeys_and_hears_what_failed():
    import json
    replies = iter([json.dumps({**DESIGNS["two storeys"], "furniture": {"bed": [0, 0, 1]}}),
                    json.dumps(DESIGNS["two storeys"])])
    seen = []

    def ask(msgs):
        seen.append(msgs[-1]["content"])
        return next(replies)
    d = architect.design("a two-storey house", ask, budget=200)
    assert d["failed"] == [] and d["spec"] == DESIGNS["two storeys"], d["failed"]
    assert "no light" in seen[1] or "no chest" in seen[1], seen[1]


ROOFS = {"gable of shingles": {"style": "gable"}, "hip of planks, stone eaves": {"style": "hip", "material": "plank", "eaves": "stone_block"},
         "gable of earth, a plank ridge": {"style": "gable", "material": "dirt", "ridge": "plank"}, "flat stone": {"material": "stone_block"}}


@pytest.mark.parametrize("roof", ROOFS)
def test_a_roof_of_any_style_and_material_is_laid_whole_by_building_alone(roof):
    """A roof is blocks, as a builder makes one: stepped up to a ridge or a peak, shingles, planks, stone or earth
    (stone and earth on a deck of planks: the game holds them by blocks only), eaves and ridge of another."""
    spec = {**DESIGNS["house"], "width": 5, "roof": ROOFS[roof]}
    s = Structure(spec)
    roof_cells = [c for c in s.cells() if c[2] >= 3]
    if ROOFS[roof].get("style", "flat") != "flat":
        assert max(z for _, _, z, _ in roof_cells) > 3                          # (it rises: layers over layers)
    for k in ("material", "eaves", "ridge"):
        assert (ROOFS[roof].get(k) or ROOFS[roof].get("material", "roof")) in {m for *_, m in roof_cells}
    b = judge3d.built(spec)
    assert b["sited"] and b["failed"] == [], b


def test_earth_goes_on_its_deck_over_walls_and_furniture_too_and_a_wrong_roof_is_told():
    """Over the whole deck: what stands on the land under it is no more in its way (nexel-worlds#316)."""
    s = Structure({**DESIGNS["house"], "roof": {"material": "dirt"}})
    dirt = {(x, y) for x, y, z, m in s.cells() if m == "dirt"}
    assert set(s.storeys[0]) <= dirt and {m for x, y, z, m in s.cells() if z == 3} == {"plank"}
    for bad, why in (({"style": "spire"}, "style is one of"), ({"material": "glass"}, "a roof is of")):
        assert why in judge3d.judged({**DESIGNS["house"], "roof": bad})["error"]


def test_the_mind_says_the_build_in_its_words_and_the_architect_draws_those(tmp_path):
    """The planner names the build in its own words (goal build, "build": ...); the architect gets them as its brief,
    with whom it is for, and turns them into a design."""
    import json
    import time
    from types import SimpleNamespace

    from nexel_agent.rl.core import SimEnv
    from nexel_agent.rl.mind import Mind
    from nexel_agent.rl.now import Now
    m = Mind("Rook", "a quiet builder", "", None, tmp_path / "memory.json", Now("Rook"))
    m.apply({"goal": "build", "build": "a round house with a sod roof"})
    assert m.building and m.build_words == "a round house with a sod roof"
    m.apply({"goal": "gather", "item": "wood", "build": "a castle"})
    assert m.build_words is None                                                    # (words for a build only)
    seen = []

    def ask(msgs):
        seen.append(msgs[0]["content"])
        return json.dumps(DESIGNS["round home"])
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    d = architect.Drafting(ask, tmp_path / "plan.json", "a quiet builder")
    social = SimpleNamespace(home=None, plan=None)
    while d.holds(social, True, env.sim.observe(env.a), None, "a round house with a sod roof"):
        time.sleep(0.01)
    assert "a round house with a sod roof. It is for: a quiet builder" in seen[0]
    assert social.plan is not None and json.loads((tmp_path / "plan.json").read_text())["spec"]["shape"] == "round"


DUG = {"underground": True, "width": 3, "depth": 3, "door": "south", "furniture": HOME}


def test_a_home_dug_underground_is_closed_and_roofed_by_the_land():
    """A trench stepping down a layer a tile to the door; the room behind it tunnelled two cells high, the land round
    it its walls and over it its roof; the furniture on its floor."""
    from nexel_agent.rl.dug import Dug
    s = Dug(DUG)
    digs = [c for c in s.cells() if c[3] == "dig"]
    assert {c[2] for c in digs if c[:2] in s.inside} == {-1, -2}                     # (the land's top left: a roof)
    assert [max(-c[2] for c in digs if c[:2] == t) for t in reversed(s.trench)] == [0, 1, 2]
    assert not any(t == "wall" for t in s.storeys[0].values()) and s.cost() == 0
    j = judge3d.judged(DUG)
    assert j["kind"] == "dug" and j["failed"] == [], j["failed"]


@pytest.mark.parametrize("spec, seed", [(DUG, 5), ({**DUG, "shape": "round", "radius": 3, "door": "east"}, 20)])
def test_a_home_is_dug_whole_by_building_alone(spec, seed):
    """Each cell from the tile dug beside it; the furniture once all is dug (a chair set early stands in the way); its
    own trench never filled as a pit; what the view cannot show, known from the game's answers."""
    b = judge3d.built(spec, seed=seed)
    assert b["sited"] and b["failed"] == [], b


GARDEN = {"garden": True, "width": 5, "depth": 4, "door": "south"}


def test_a_garden_is_fenced_with_a_gate_and_its_beds_each_beside_a_walk():
    from nexel_agent.rl.garden import Garden
    g = Garden(GARDEN)
    near = lambda p: [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)]   # noqa: E731
    assert all(any(q in g.walks or q in g.ring for q in near(b)) for b in g.beds)     # (each bed reached)
    assert list(g.storeys[0].values()).count("gate") == 1 and g.cost() == len(g.ring)
    cells = g.cells()
    assert all(cells.index((*b, 0, "till")) < cells.index((*b, 0, "seed")) for b in g.beds)
    assert judge3d.judged(GARDEN)["failed"] == []
    assert "beds are" in judge3d.judged({**GARDEN, "beds": "maze"})["error"]


@pytest.mark.parametrize("spec", [GARDEN, {"garden": True, "shape": "round", "radius": 3, "door": "east"},
                                  {"garden": True, "width": 3, "depth": 3, "door": "north", "beds": "full"}])
def test_a_garden_is_fenced_tilled_and_sown_by_building_alone(spec):
    """The fence and gate laid, each bed tilled with the hoe from the pack, then sown: the seed only takes on soil."""
    b = judge3d.built(spec, seed=20)
    assert b["sited"] and b["failed"] == [], b


COMPOUNDS = {
    "house and garden": {"parts": [{**DESIGNS["house"], "roof": {"style": "gable"}}, {**GARDEN, "at": "east"}]},
    "dug home and its beds": {"parts": [DUG, {"garden": True, "width": 3, "depth": 3, "door": "north", "beds": "full",
                                              "at": "north"}]},
    "two storeys and a walled yard": {"parts": [DESIGNS["two storeys"], {**DESIGNS["round base"], "at": "west"}]},
}


def test_parts_are_set_apart_on_their_sides_and_judged_each_as_itself():
    from nexel_agent.rl.compound import GAP, Compound
    c = Compound(COMPOUNDS["house and garden"])
    house, garden = (c._moved(k, p.clear()) for k, p in enumerate(c.parts))
    assert min(x for x, _ in garden) - max(x for x, _ in house) > GAP                  # (east of it, a gap between)
    j = judge3d.judged(COMPOUNDS["house and garden"])
    assert j["kind"] == "home+garden" and j["failed"] == [] and len(j["parts"]) == 2
    assert "1 to 4" in judge3d.judged({"parts": [GARDEN] * 5})["error"]


@pytest.mark.parametrize("name, seed", [("house and garden", 40), ("dug home and its beds", 20),
                                        ("two storeys and a walled yard", 60)])
def test_a_compound_longer_than_a_view_is_laid_whole_by_building_alone(name, seed):
    """A part at the far end out of view is walked to; what was laid at the other end is known as it was last seen;
    a roof waits for its own part's walls only; the land between the parts is the land's (no pit to fill)."""
    b = judge3d.built(COMPOUNDS[name], seed=seed)
    assert b["sited"] and b["failed"] == [], b


def test_no_build_is_sited_where_water_runs():
    """The game's rivers and lakes run onto flat land round them: nothing is sited within WET tiles of water."""
    from nexel_agent.rl.core import SimEnv
    from nexel_agent.rl.site import WET, _near_water
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    a = env.a
    env.sim.world.set(a.x + 3, a.y, ground="water")
    wet = _near_water(env.sim.observe(a))
    assert (a.x + 3 + WET, a.y) in wet and (a.x + 4 + WET, a.y) not in wet


def test_the_mind_adds_a_garden_beside_its_home_once_it_stands(tmp_path):
    """The home drawn and laid; then, the mind naming another build in its words, that part is drawn beside it on the
    same site and laid, and the whole stands, each part judged as itself."""
    import json
    import time
    from collections import Counter

    from nexel_agent.rl.actions import ACTIONS
    from nexel_agent.rl.core import SimEnv
    from nexel_agent.rl.dug import design_of
    replies = iter([json.dumps(DESIGNS["house"]), json.dumps({**GARDEN, "at": "east"})])
    d = architect.Drafting(lambda msgs: next(replies), tmp_path / "plan.json", "a farmer")
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(20)
    env.folk = []
    whole = design_of({"parts": [DESIGNS["house"], {**GARDEN, "at": "east"}]})
    env.a.inventory.clear()
    env.a.inventory.update(Counter("stone_wall" if p == "wall" else p for *_, p in whole.cells() if p != "till"))
    env.a.inventory.update({"hoe": 1})
    b = next(i for i, a in enumerate(ACTIONS) if a[0] == "build")

    def build(words):
        for _ in range(400):
            while d.holds(env.social, True, env.sim.observe(env.a), None, words):
                time.sleep(0.01)
            if not env.encoded()["mask"][b]:
                return
            env.step(b)
    build("a stone house")
    home = env.social.home
    assert home is not None and json.loads((tmp_path / "plan.json").read_text())["spec"] == DESIGNS["house"]
    build("a vegetable garden by the house")
    assert env.social.home == home                                                   # (the same site: it grew)
    kept = json.loads((tmp_path / "plan.json").read_text())["spec"]
    assert [p.get("garden", False) for p in kept["parts"]] == [False, True]
    gx, gy = env.social.plan.ground[0]
    j = judge3d.judge_at(env.sim, design_of(kept), *home, env.sim.world.level(home[0] + gx, home[1] + gy))
    assert j["failed"] == [], j["failed"]


def test_the_mind_reads_a_drawn_build_by_what_is_done_and_counts_it_built_when_whole(tmp_path):
    """A dug room seen on the land's view is no room the judge can read: a drawn build is told by its pieces done and
    what is left, and it is built when every piece is done."""
    from collections import Counter

    from nexel_agent.rl import home as H
    from nexel_agent.rl.actions import ACTIONS
    from nexel_agent.rl.core import SimEnv
    from nexel_agent.rl.dug import design_of
    from nexel_agent.rl.mind import Mind
    from nexel_agent.rl.now import Now
    from nexel_agent.rl.plan import of_structure
    s = design_of(DUG)
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(20)
    env.folk = []
    env.social.plan = of_structure(s)
    env.a.inventory.clear()
    env.a.inventory.update(Counter(p for *_, p in s.cells() if p != "dig"))
    env.a.inventory.update({"pickaxe": 1})
    b = next(i for i, a in enumerate(ACTIONS) if a[0] == "build")
    env.step(b)
    r = H.reading(env.sim.observe(env.a), env.social.home, env.social)
    assert 0 < r["pieces"][0] < r["pieces"][1] and "dig" in r["left"]
    assert "pieces done; left:" in H.told(env.sim.observe(env.a), r=r)
    for _ in range(300):
        if not env.encoded()["mask"][b]:
            break
        env.step(b)
    obs = env.sim.observe(env.a)
    r = H.reading(obs, env.social.home, env.social)
    assert r["pieces"][0] == r["pieces"][1] and "whole" in H.told(obs, r=r)
    m = Mind("Rook", "", "", None, tmp_path / "memory.json", Now("Rook"))
    m.home, m.home_read = env.social.home, r
    assert m.stands() == ("it stands whole", True)


def test_with_no_spot_in_view_it_looks_further_and_clears_only_what_grows():
    """A drawn build with nowhere to stand in view walks a view on and looks again (the hut, the body's own, does
    not); a site may hold trees and flowers to clear, never what someone made."""
    from types import SimpleNamespace

    from nexel_agent.rl.build import CLEARABLE, CH, HUT, _look_further
    from nexel_agent.rl.core import SimEnv
    from nexel_agent.rl.plan import of_structure
    assert CH["tree"] in CLEARABLE and CH["flower"] in CLEARABLE
    assert not {CH[k] for k in ("chest", "bed", "crop", "workbench", "stone_wall")} & CLEARABLE
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    obs, social = env.sim.observe(env.a), SimpleNamespace(home=None)
    assert _look_further(obs, social, HUT) is None
    tool, args = _look_further(obs, social, of_structure(Structure(DESIGNS["house"])))
    assert tool == "move_to" and max(abs(args["x"] - env.a.x), abs(args["y"] - env.a.y)) >= 9
    assert _look_further(obs, social, of_structure(Structure(DESIGNS["house"])))[1] == args     # (same tick: the same)


@pytest.mark.parametrize("name, seed", [("tower", 44), ("round home", 23)])
def test_a_build_with_no_spot_where_it_starts_finds_one_further_and_is_laid_whole(name, seed):
    b = judge3d.built(DESIGNS[name], seed=seed)
    assert b["sited"] and b["failed"] == [], b


def test_the_planner_is_told_every_kind_it_can_build():
    """What the architect and the build verb can lay is what the planner is told it may name: else it asks for huts."""
    from nexel_agent.rl.mind import GOALS_DOC
    for kind in ("tower", "round", "underground", "garden", "gabled", "earth", "beside it"):
        assert kind in GOALS_DOC, kind


def test_the_mind_is_told_what_its_build_is_in_a_few_words():
    from nexel_agent.rl.compound import describe
    assert describe(DESIGNS["tower"]) == "a three-storey tower"
    assert describe(DESIGNS["round base"]) == "a round walled yard"
    assert describe({**DESIGNS["house"], "roof": {"style": "gable", "material": "dirt"}}) == "a house, a gable roof of earth"
    assert describe(COMPOUNDS["house and garden"]) == "a house, a gable roof of shingles; a garden east"
    plan = architect.plan_from({"spec": COMPOUNDS["house and garden"]})
    assert plan.words == describe(COMPOUNDS["house and garden"])
    from nexel_agent.rl import home as H
    r = {"far": False, "where": "3 tiles N", "pieces": [10, 90], "left": {"fence": 4}, "lacks": {}, "what": plan.words}
    assert H.told({}, r=r).startswith("Your build 3 tiles N (a house, a gable roof of shingles; a garden east): 10 of 90")


def test_over_budget_the_architect_is_told_what_costs_most_and_any_design_is_read():
    import json
    big = {**DESIGNS["two storeys"], "width": 5, "depth": 5, "roof": {"style": "gable"}}
    j = architect.judged(json.dumps(big), 82)
    assert "cost" in j["failed"] and "roof" in j["told"] and "walls" in j["told"] and "flat roof" in j["told"]
    assert architect.spec_of(json.dumps({"garden": True})) == {"garden": True}                # (read as a design...)
    assert "width and depth" in architect.judged('{"garden": true}', 82)["told"]               # (...and told what it lacks)
    assert architect.judged(json.dumps({**DESIGNS["house"], "roof": {"style": "gable"}}), architect.BUDGET)["failed"] == []


def test_eaves_are_the_roofs_outer_ring_whatever_its_style():
    """Bench run 9: on a flat roof every tile was at depth 0, so "eaves" of stone took it all, and stone is held by no
    wall: none of it went down. The eaves are the outer ring only; and the best design is one within the budget."""
    s = Structure({**DESIGNS["house"], "roof": {"style": "flat", "eaves": "stone_block"}})
    roof = [(x, y, m) for x, y, z, m in s.cells() if z == 3]
    xs, ys = [x for x, _, _ in roof], [y for _, y, _ in roof]
    edge = lambda x, y: x in (min(xs), max(xs)) or y in (min(ys), max(ys))          # noqa: E731
    assert all((m == "stone_block") == edge(x, y) for x, y, m in roof)
    b = judge3d.built({**DESIGNS["house"], "roof": {"style": "flat", "eaves": "stone_block"}}, seed=5)
    assert b["failed"] == [], b
    assert architect._rank({"ok": True, "failed": ["cost"], "grade": 22}) < \
        architect._rank({"ok": True, "failed": ["v2.chest"], "grade": 18})
    assert architect._rank({"ok": False, "failed": [], "grade": 0}) < architect._rank({"ok": True, "failed": ["cost"], "grade": 16})


def test_the_last_piece_under_its_feet_is_stepped_off_and_laid():
    """Bench run 11: a garden's path stood unfinished - its last tile was under the builder's feet (the view shows
    the builder there, not the ground), so it was skipped and nothing was left to do; and stepping off onto the other
    path tile, the first was forgotten. Now it steps off to lay it, and what it saw laid under its feet stays laid."""
    spec = {"garden": True, "door": "south", "beds": "rows", "path": 2, "width": 5, "depth": 3}
    b = judge3d.built(spec, seed=20)
    assert b["failed"] == [] and b["picks"] < 100, b
