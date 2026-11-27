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


def test_earth_goes_only_over_bare_land_and_a_wrong_roof_is_told():
    s = Structure({**DESIGNS["house"], "roof": {"material": "dirt"}})
    busy = set(s.storeys[0]) | set(s.path)
    assert all(m != "dirt" for x, y, z, m in s.cells() if z >= 3 and (x, y) in busy)
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
