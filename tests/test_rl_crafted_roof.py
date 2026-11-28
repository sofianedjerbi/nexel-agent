"""A craftsman's roof (the owner, 2026-11-28: "roof can be artisanal made if different blocks, it isn't only the roof
block"): a design whose roof is one material seen from above fails "crafted_roof", and is told how to mix it."""

from test_rl_structure import DESIGNS

from nexel_agent.rl import architect, judge3d


def test_a_roof_of_one_material_fails_and_one_mixed_passes():
    house = DESIGNS["house"]
    for one in ({"style": "flat", "material": "dirt", "eaves": "dirt"}, {"style": "gable"}, {"material": "stone_block"}):
        j = judge3d.judged({**house, "roof": one})
        assert "crafted_roof" in j["failed"], (one, j["failed"])
        assert "craftsman's roof mixes them" in judge3d.told(j)
    for mixed in ({"style": "flat", "material": "stone_block", "eaves": "plank"}, {"style": "gable", "ridge": "stone_block"},
                  {"style": "hip", "material": "plank", "eaves": "dirt"}):
        assert "crafted_roof" not in judge3d.judged({**house, "roof": mixed})["failed"], mixed


def test_an_open_yard_has_no_roof_to_craft():
    assert "crafted_roof" not in judge3d.judged(DESIGNS["round base"])["failed"]


def test_the_architect_is_told_to_mix_and_redraws_until_it_does():
    import json
    replies = iter([json.dumps({**DESIGNS["house"], "roof": {"material": "dirt"}}),
                    json.dumps({**DESIGNS["house"], "roof": {"material": "dirt", "eaves": "plank"}})])
    seen = []
    d = architect.design("a home", lambda m: seen.append(m[-1]["content"]) or next(replies))
    assert d["failed"] == [] and len(d["history"]) == 2, d["history"]
    assert "craftsman's roof mixes them" in seen[1]
