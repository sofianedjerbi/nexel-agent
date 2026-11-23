"""rl/facts.py: the one reader of the game's events, the one distance."""

import math

from nexel_agent.rl import facts as F
from nexel_agent.rl.social import SocialMemory
from nexel_agent.rl.voice import Now


def test_one_record_per_event_type():
    assert F.read({"type": "attack", "attacker": "a tusker", "target": "Rook", "damage": 2, "id": 7, "tick": 5}) == \
        F.Attack("a tusker", "Rook", 2, None, 7, 5)
    assert F.read({"type": "give", "from": "Mira", "to": "Rook", "items": {"stone": 2}}) == F.Give("Mira", "Rook", {"stone": 2})
    assert F.read({"type": "shout", "from": "Mira", "text": "hey"}) == F.Say("Mira", "hey", True)
    assert F.read({"type": "death", "name": "Rook", "cause": "killed by Bob", "by": "Bob"}) == F.Death("Rook", "killed by Bob")
    assert F.read({"type": "creature_died", "kind": "tusker", "by": "Rook", "id": 7}) == F.Fell("tusker", "Rook", 7)
    assert F.read({"type": "tile"}) is None
    assert [type(ev) for ev in F.events({"events": [{"type": "join", "name": "Mira"}, {"type": "xp"}, {"type": "boss", "what": "wake"}]})] \
        == [F.Join, F.Boss]
    assert set(F.PARSERS) == {"say", "shout", "attack", "give", "death", "creature_died", "craft", "join", "boss", "pick_up"}
    assert F.read({"type": "pick_up", "name": "Rook", "items": {"wood": 2}, "x": 3, "y": 4}) == F.Took("Rook", {"wood": 2}, None, (3, 4))
    assert F.read({"type": "death", "name": "Mira", "x": 3, "y": 4, "dropped": {"wood": 1}}) == F.Death("Mira", None, None, (3, 4), {"wood": 1})


def test_an_event_is_worded_once():
    you = Now("Rook").who
    assert F.told(F.read({"type": "death", "name": "Rook", "cause": "killed by Bob", "by": "Bob"})) == "Rook died (killed by Bob)"
    assert F.told(F.read({"type": "give", "from": "Mira", "to": "Rook", "items": {"stone_block": 2}}), you) == "Mira gave you 2 stone block"
    assert F.told(F.read({"type": "attack", "attacker": "Rook", "target": "tusker#7"}), you) == "you hit a tusker"
    assert F.told(F.read({"type": "creature_died", "kind": "tusker", "by": "Rook"}), you) == "a tusker fell to you"


def test_a_gift_is_remembered_as_what_it_was():
    now = Now("Rook")
    now.see({"self": {}, "events": [{"type": "give", "from": "Mira", "to": "Rook", "items": {"berries": 3}}]})
    assert now.relations["Mira"]["record"] == {"gave you berries": 3}


def test_the_items_a_line_names():
    items = ("ore", "stone", "stone_block", "wood", "plank")
    assert F.named("got any more?", items) == []                          # ("more" is not "ore")
    assert F.named("could you spare some stone_block?", items) == ["stone_block", "stone"]
    assert F.named("Two PLANKS and some wood pls", items) == ["plank", "wood"]
    social = SocialMemory("Rook", items)
    social.see({"tick": 9, "events": [{"type": "say", "from": "Mira", "text": "got any more?"}]})
    assert social.said_to_me == 9 and social.ask_open(9) is None
    social.see({"tick": 12, "events": [{"type": "say", "from": "Mira", "text": "any stone block left?", "tick": 11}]})
    assert (social.ask_open(12), social.asked_by, social.asked_at) == ("stone_block", "Mira", 11)


def test_one_distance_the_servers():
    b = {"dx": 3, "dy": 1}
    assert F.dist(b) == math.hypot(3, 1)
    assert F.within(b, 3) and not F.within({"dx": 3, "dy": 2}, 3)       # (sim/bodies.py near: dx2 + dy2 <= (r + .5)2)
    assert F.compass(3, 3) == "4 tiles SE" and F.compass(0, -6) == "6 tiles N" and F.compass(1, 1) == "right here"
    obs = {"entities": [{"kind": "tusker", "dx": 1, "dy": 0}, {"kind": "agent", "name": "Far", "dx": 4, "dy": 4},
                        {"kind": "agent", "name": "Near", "dx": 5, "dy": 0}]}
    assert [p["name"] for p in F.people(obs)] == ["Near", "Far"]
    assert F.is_creature("a tusker") and F.is_creature("tusker#7") and not F.is_creature("Mira")
