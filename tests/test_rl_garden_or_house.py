"""A house drawn with "garden": true (live trial 12, rl_runs/score/20261128-155737: the model filled every key, its
"small garden outside" a flag on the house): read as a garden, its roof, storey and furniture were dropped and a pen
was laid. A garden with a roof, storeys or furniture is refused and the architect told to draw it as two parts."""

import json

from nexel_agent.rl import architect

LIVE12 = {"shape": "round", "radius": 4, "storeys": 2, "door": "east",
          "roof": {"style": "gable", "material": "roof", "eaves": "plank", "ridge": "stone_block"},
          "furniture": {"bed": [[2, 2]], "campfire": [[3, 3]], "chest": [[1, 1]], "table": [[2, 4]], "chair": [[3, 4]],
                        "workbench": [[4, 3]]},
          "path": 3, "garden": True, "beds": "rows", "underground": False, "walled": False, "parts": []}


def test_a_house_flagged_a_garden_is_refused_and_told_to_be_two_parts():
    j = architect.judged(json.dumps(LIVE12), architect.NAMED)
    assert not j["ok"] and '"parts"' in j["told"] and "a garden has no roof" in j["told"], j


def test_a_garden_alone_and_a_house_beside_its_garden_still_stand():
    garden = {"garden": True, "width": 3, "depth": 3, "door": "south"}
    assert architect.judged(json.dumps(garden), architect.NAMED)["ok"]
    house = {k: v for k, v in LIVE12.items() if k not in ("garden", "beds", "underground", "walled", "parts")}
    j = architect.judged(json.dumps({"parts": [{**house, "storeys": 1, "radius": 3}, {**garden, "at": "west"}]}),
                         architect.NAMED)
    assert j["ok"] and j["kind"] == "home+garden", j
