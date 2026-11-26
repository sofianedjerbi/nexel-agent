"""rl/facts.py: the one reader of the game's events, the one distance."""

import math

from nexel_agent.rl import facts as F
from nexel_agent.rl.social import SocialMemory
from nexel_agent.rl.voice import Now


def test_one_record_per_event_type():
    assert F.read({"type": "attack", "attacker": "a tusker", "target": "Rook", "damage": 2, "id": 7, "tick": 5}) == \
        F.Attack("a tusker", "Rook", 2, None, 7, 5, "tusker", "agent")
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
    assert F.compass(3, 3) == "4 tiles SE" and F.compass(0, -6) == "6 tiles N" and F.compass(1, 1) == "right here"
    obs = {"entities": [{"kind": "tusker", "dx": 1, "dy": 0}, {"kind": "agent", "name": "Far", "dx": 4, "dy": 4},
                        {"kind": "agent", "name": "Near", "dx": 5, "dy": 0}]}
    assert [p["name"] for p in F.people(obs)] == ["Near", "Far"]
    assert [F.kind_of(None, n) for n in ("a tusker", "tusker#7", "Mira", None)] == ["tusker", "tusker", "agent", None]   # (an older game's words)


def test_a_pile_on_the_ground_is_seen_by_the_body_and_the_words():
    from nexel_agent.rl.core import SimEnv
    from nexel_agent.rl.encode import CHAR_ID, encode
    env = SimEnv(ep_ticks=200)
    env.reset(7)
    sim, a = env.sim, env.a
    x, y = next((a.x + dx, a.y + dy) for dx in range(2, 8) for dy in range(-3, 4) if sim.world.block(a.x + dx, a.y + dy) == "none")
    sim.add_items(x, y, {"meat": 2})
    obs = sim.observe(a)
    assert F.ground(obs) == [(x - a.x, y - a.y, {"meat": 2})]
    assert (encode(obs)["grid"] == CHAR_ID["$"]).sum() == 1              # (the body sees a pile, not what it is)
    now = Now(a.name)
    now.see(obs)
    assert f"2 meat on the ground {F.compass(x - a.x, y - a.y)}" in now.text()


def test_the_mask_reach_is_the_games():
    """A blow is legal where the server accepts it: the game's own rule (sim/bodies.py reach_of, in_reach), with no
    KeyError when a being is in view (the reach table is keyed by kind of action now, not by weapon)."""
    from nexel_worlds.sim.bodies import reach_of
    from nexel_agent.rl.actions import ACTIONS, legal
    hit = ACTIONS.index(("attack", None, None))
    obs = lambda dx, inv: {"self": {"x": 0, "y": 0, "hp": 20, "hunger": 20, "inventory": inv}, "grid": ["." * 21] * 21,
                           "entities": [{"id": 1, "kind": "chicken", "dx": dx, "dy": 0, "hp": 3}], "tick": 0}
    r = reach_of("melee")
    assert legal(obs(int(r), {}))[hit] and not legal(obs(int(r) + 1, {}))[hit]
    assert legal(obs(int(r) + 1, {"spear": 1}))[hit]


def test_who_did_it_is_read_by_kind_not_by_name():
    """(#35) An event names who did it as {kind, id, name} (nexel-worlds#119): a person is told from a creature by its
    kind, a death by its code. The records keep the names the readers key on (rl/social.py, rl/folk.py), the same as
    from an older game's words, which are still read when who and whom are not there (the agent's pin, 31963d92)."""
    old = {"type": "attack", "attacker": "a tusker", "target": "Rook", "damage": 2, "id": 7, "tick": 5}
    new = {**old, "who": {"kind": "tusker", "id": 7, "name": None}, "whom": {"kind": "agent", "id": 1, "name": "Rook"}}
    assert F.read(new) == F.read(old) == F.Attack("a tusker", "Rook", 2, None, 7, 5, "tusker", "agent")
    named = F.read({"type": "attack", "who": {"kind": "agent", "id": 3, "name": "a_tusker"}, "whom": {"kind": "tusker", "id": 9, "name": None}})
    assert named.by == "a_tusker" and named.by_person and named.target == "tusker#9" and not named.on_person   # (a player named a_tusker is a person)
    died = F.read({"type": "death", "name": "Rook", "cause": "tusker", "code": "bitten", "who": {"kind": "tusker", "id": 7, "name": None}})
    assert died.code == "bitten" and died.cause == "tusker"


def test_the_sims_own_blows_read_the_same_with_who_and_whom():
    """(#35, t8) The training's sim (the pin's game: words only) and a game that names who did it give the same
    records: a person's blow on a person (the rival's), a person's on a creature, a creature's on a person."""
    from nexel_agent.rl.core import SimEnv
    from nexel_worlds.sim.entities import CREATURES, Creature
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    env.reset(5)
    sim, a, rv = env.sim, env.a, env.rival
    rv.next = sim.tick + 1
    env.step(0)
    rv.a.x, rv.a.y, rv.a.z = a.x - 2, a.y, 99                     # (as test_rl_people's attack back)
    c = Creature(sim.next_creature, "tusker", a.x + 1, a.y, CREATURES["tusker"]["hp"])
    sim.next_creature += 1
    sim.creatures[c.id] = c
    sim.submit(rv.a, "attack", {"target": a.id})
    sim.submit(a, "attack", {"target": c.id})
    seen = []
    for _ in range(6):
        env.step(0)
        seen += [e for e in env.obs.get("events", []) if e.get("type") == "attack"]
    people = {p.name: p.id for p in sim.agents.values()}
    ref = lambda name, blow: {"kind": "agent", "id": people[name], "name": name} if name in people else \
        {"kind": name[2:] if blow else name.split("#")[0], "id": int(name.split("#")[1]) if "#" in name else c.id, "name": None}
    pairs = {(F.read(e).by_kind, F.read(e).target_kind) for e in seen}
    assert ("agent", "agent") in pairs and ("agent", "tusker") in pairs, seen
    for e in seen:
        assert F.read({**e, "who": ref(e["attacker"], True), "whom": ref(e["target"], False)}) == F.read(e), e


def test_a_death_is_told_from_its_code_and_who_did_it():
    """(nexel-worlds#171: no cause words) A death in words from its code and who, as the game's death screen says it;
    an older game's words when there is no code."""
    you = Now("Rook").who
    bob, tusker = {"kind": "agent", "id": 2, "name": "Bob"}, {"kind": "tusker", "id": 7, "name": None}
    assert F.told(F.read({"type": "death", "name": "Rook", "code": "killed", "who": bob}), you) == "you died (killed by Bob)"
    assert F.told(F.read({"type": "death", "name": "Mira", "code": "bitten", "who": tusker})) == "Mira died (bitten by a tusker)"
    assert F.told(F.read({"type": "death", "name": "Mira", "code": "starvation"})) == "Mira died (starved)"
    assert F.told(F.read({"type": "death", "name": "Mira", "cause": "killed by Bob"})) == "Mira died (killed by Bob)"
