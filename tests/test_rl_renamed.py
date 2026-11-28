"""The game renames a thing (nexel-worlds#403: the sword is the iron sword, the ore and the ingot the iron ones): the
body's frozen rows and slots keep their places and their meaning, read and sent under the game's new names (#94)."""

from nexel_worlds.ores import RENAMED
from nexel_worlds.tiles import RECIPES

from nexel_agent.rl import actions, encode, vocab


def test_a_learned_craft_of_a_renamed_thing_crafts_it_under_its_new_name():
    for old, new in RENAMED.items():
        if ("craft", None, old) not in vocab.ACTIONS:
            continue
        i = vocab.ACTIONS.index(("craft", None, old))
        assert actions.ACTIONS[i] == ("craft", None, old)                  # (the row stays where it was learned)
        assert actions.to_command(i, {"self": {"x": 0, "y": 0, "inventory": {}}}, "S") == ("craft", {"recipe": new})
        assert new in RECIPES


def test_a_slot_reads_the_renamed_thing_and_a_goal_names_it_as_the_game_does():
    from nexel_agent.rl.core import SimEnv
    env = SimEnv(roam=0, rivals=0)
    env.reset(9)
    env.a.inventory["iron_sword"] = 1
    v = encode.encode(env.sim.observe(env.a))["vec"]
    assert v[encode.BODY + vocab.ITEMS.index("sword")] > 0
    assert encode.ITEM_ID["iron_ore"] == encode.ITEM_ID["ore"] and encode.ITEM_ID["iron_sword"] == encode.ITEM_ID["sword"]
    assert vocab.now("craft_ingot") == "craft_iron_ingot" and vocab.now("stone_block") == "stone_block"
