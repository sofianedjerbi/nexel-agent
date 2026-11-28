"""A base: a compound's parts closed in by one wall of stone with a gate (rl/rampart.py), laid last, judged on its
own (closed, a gate), the parts each as themselves."""

from nexel_agent.rl import judge3d
from nexel_agent.rl.compound import Compound, describe

HOUSE = {"width": 3, "depth": 3, "door": "south", "furniture": {"bed": [0, 0], "campfire": [2, 0], "chest": [0, 2], "chair": [2, 2]},
         "roof": {"style": "gable", "eaves": "stone_block", "ridge": "plank"}}
BASE = {"parts": [HOUSE, {"garden": True, "width": 5, "depth": 3, "at": "east"},
                  {"shape": "round", "radius": 3, "door": "south", "storeys": 2, "at": "west",
                   "furniture": {"campfire": [3, 3], "bed": [1, 2], "chest": [3, 1], "table": [1, 1, 1]},
                   "roof": {"style": "hip", "material": "plank", "eaves": "stone_block"}}],
        "walled": "south"}


def test_a_base_walled_round_is_judged_whole_and_laid_whole_by_building_alone():
    j = judge3d.judged(BASE)
    assert j["failed"] == [] and j["kind"] == "home+garden+home+wall", j["told"]
    b = judge3d.built(BASE, seed=7)
    assert b["sited"] and b["failed"] == [], b


def test_the_wall_goes_round_every_part_last_with_its_gate_and_is_told():
    c = Compound(BASE)
    wall = c.parts[-1]
    inner = set().union(*(c._moved(k, p.clear()) for k, p in enumerate(c.parts[:-1])))
    xs, ys = [x for x, _ in wall.ring], [y for _, y in wall.ring]
    assert all(min(xs) < x < max(xs) and min(ys) < y < max(ys) for x, y in inner)     # (everything inside it)
    cells = c.cells()
    assert {p for *_, p in cells[-len(wall.cells()):]} == {"wall", "door", "path"}      # (laid after every part)
    assert wall.doors[0][1] == max(ys) and describe(BASE).endswith("a wall round them all")
    j = judge3d.judged({**BASE, "parts": BASE["parts"][:1]})
    assert j["failed"] == [], j["told"]
