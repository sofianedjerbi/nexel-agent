"""The words for a build, read for what they ask plainly, and a design judged on keeping it (rl/asked.py)."""

import json

from nexel_agent.rl import architect
from nexel_agent.rl.asked import asked, unlike

from test_rl_structure import DESIGNS


def test_the_words_ask_a_roof_a_round_and_storeys_and_nothing_they_leave_out():
    assert asked("a small home with 1 room, 1 bed, 1 door, and a gabled roof made of shingles") == {"roof": "gable"}
    assert asked("a round house with a domed roof") == {"roof": "hip", "shape": "round"}
    assert asked("a two-storey house with a flat roof") == {"roof": "flat", "storeys": 2}
    assert asked("a flat or gabled roof") == {}                     # (either will do)
    assert asked("a home for one") == {} and asked("a flat stone path; a roof of planks") == {}


def test_a_design_unlike_the_words_fails_them_and_is_told_how():
    """Live trial 4: "a gabled roof made of shingles" asked, a flat roof drawn at full marks."""
    words = "a small home with a gabled roof made of shingles"
    flat = json.dumps({**DESIGNS["house"], "roof": {"style": "flat", "eaves": "plank"}})
    j = architect.judged(flat, architect.NAMED, asked(words))
    assert j["failed"] == ["words"] and "the words ask for a gable roof, this one is flat" in j["told"]
    good = architect.judged(json.dumps({**DESIGNS["house"], "roof": {"style": "gable", "ridge": "plank"}}), architect.NAMED, asked(words))
    assert good["failed"] == [] and good["grade"] == good["of"] == j["of"]
    assert unlike({**DESIGNS["house"], "garden": True}, {"roof": "gable"}) == ""          # (a garden: not read here)


def test_the_architect_keeps_redrawing_until_the_words_are_kept():
    replies = iter([json.dumps({**DESIGNS["house"], "roof": {"style": "flat", "eaves": "plank"}}),
                    json.dumps({**DESIGNS["house"], "roof": {"style": "gable", "ridge": "plank"}})])
    d = architect.design("a home. What they carry now: 12 roof", lambda m: next(replies), words="a home under a gable roof")
    assert d["failed"] == [] and len(d["history"]) == 2 and json.loads(json.dumps(d["spec"]))["roof"]["style"] == "gable"
