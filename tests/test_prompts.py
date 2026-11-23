"""The prompts tell the AI players what the game's blocks are (moved here from nexel-worlds tests/test_land_blocks.py)."""

from nexel_agent.prompts import RULES, recipes_text, tools_text


def test_the_agents_are_told():
    tools, book = tools_text(), recipes_text()
    assert "stone_block" in tools and "not a material" in tools and "stone_block" in RULES
    assert "stone = 1 stone_block" in book
