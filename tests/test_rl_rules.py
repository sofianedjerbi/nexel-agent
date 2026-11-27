"""The RL env's legal mask, live-seed starts, social achievements and goals."""

import random

import numpy as np

from nexel_agent.rl.achieve import ACHIEVEMENTS, Ledger, crafter_score
from nexel_agent.rl.core import HURT, SimEnv
from rl_kit import during_next_tick, idx, neighbour_beside, to_command_for
from nexel_agent.rl.encode import ACTIONS, N_ACTIONS, N_CHARS, SIDE, VEC, action_name, legal
from nexel_agent.rl import folk


def test_legal_mask_follows_the_rules():
    from nexel_agent.rl.encode import legal
    env = SimEnv(ep_ticks=10 ** 6, roam=0)
    o = env.reset(5)
    assert o["mask"].shape == (N_ACTIONS,) and o["mask"][0]           # noop is always there
    env.a.inventory.update({"wood": 4, "stone": 3, "fiber": 1, "berries": 1})
    env.a.hunger = 10                                                 # (hungry: food can be eaten)
    m = legal(env.sim.observe(env.a))
    assert m[idx("craft_workbench")] and m[idx("eat_berries")] and not m[idx("eat_meat")]
    assert not m[idx("craft_axe")]                                    # (no workbench within reach)
    assert not m[idx("dig_N_axe")] and m[idx("dig_N")]
    if not env.sim.observe(env.a).get("heights"):                     # (flat all round: nothing in front that high)
        assert not m[idx("dig_at_N_1")] and not m[idx("dig_at_N_2")]


def test_live_seed_share_and_roaming_start():
    env = SimEnv(ep_ticks=100, live_seed=12, live_pct=100)
    env.reset(3)
    assert env.sim.seed == 12 and env.a.online
    spawn = SimEnv(ep_ticks=100, roam=0, rivals=0)
    spawn.reset(3)
    roam = SimEnv(ep_ticks=100, rivals=0)
    roam.reset(3)
    assert (spawn.a.x, spawn.a.y) != (roam.a.x, roam.a.y) and roam.rival is None


def test_social_once_each_and_helping():
    from nexel_agent.rl.social import SOCIAL
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a, f = env.sim, env.a, env.folk[0]
    f.next = sim.tick + 1
    env.step(0)
    assert f.on
    f.act = 10 ** 9                                   # (it stands still and does nothing more by itself)
    f.a.x, f.a.y, f.a.z = a.x + 1, a.y, 99
    a.inventory["wood"] = 3
    sim.submit(f.a, "say", {"text": "got any wood to spare?"})
    o, *_ = env.step(0)
    from nexel_agent.rl.encode import ACTIONS_V1, BODY, ITEMS, KINDS, N_NEAR
    s = BODY + len(ITEMS) + N_NEAR * (3 + len(KINDS)) + ACTIONS_V1
    assert o["vec"][s] == 1 and o["vec"][s + 1] == 1    # (spoken to; asked for something it carries)
    env.step(idx("give"))
    env.step(0)
    assert {"gave_item", "helped"} <= env.ledger.done and f.a.count("wood") >= 1
    env.step(idx("say"))
    env.step(0)
    assert "answered_someone" in env.ledger.done
    before = set(env.ledger.done) & set(SOCIAL)
    for _ in range(4):                                # (talking more earns nothing more)
        env.step(idx("say"))
    assert set(env.ledger.done) & set(SOCIAL) <= before | {"greeted_someone"} and len(SOCIAL) == 6


def test_live_body_imports():
    import nexel_agent.rl.live as live
    assert live.LiveBody and live.split


def test_refusals_the_obs_shows_are_masked():
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    env.reset(5)
    a = env.a
    env.sim.set_tile(a.x + 1, a.y, block="none")
    a.inventory.update({"hoe": 1, "berries": 1})
    a.hunger = 20
    m = legal(env.sim.observe(a), "E")
    assert not m[idx("dig_E")] and not m[idx("dig_E_axe")] and m[idx("dig_E_hoe")]    # (bare level ground: tilling)
    assert not m[idx("eat_berries")]                                                   # (full)
    env.sim.set_tile(a.x + 1, a.y, block="tree")
    m = legal(env.sim.observe(a), "E")
    assert m[idx("dig_E")] and not m[idx("dig_E_hoe")] and not m[idx("step_E")]


def test_it_sees_what_it_has_already_done():
    from nexel_agent.rl.vocab import KNOWN          # (the encoder's own, frozen numbering)
    from nexel_agent.rl.encode import DONE_ID, V3
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    o = env.reset(5)
    bits = lambda o: {KNOWN[i] for i in np.flatnonzero(o["vec"][V3:V3 + len(KNOWN)])}   # noqa: E731
    assert bits(o) == set()
    env.a.inventory["wood"] = 6
    o, r, *_ = env.step(idx("craft_plank"))
    assert r >= 1.0 and "craft_plank" in bits(o)                 # (the first: paid, and now seen as done)
    o, r, *_ = env.step(idx("craft_plank"))
    assert r < 1.0 and bits(o) == env.ledger.done & set(DONE_ID)  # (the repeat pays nothing: the same record says so)
    assert "shelter" not in DONE_ID and "greeted_someone" not in DONE_ID    # (what takes a look at the world or at others)


def test_the_live_body_keeps_the_same_record():
    from nexel_agent.rl.achieve import from_wire
    sim, live = Ledger(), Ledger()
    script = [("craft", {"recipe": "plank"}, {"ok": True, "made": {"plank": 2}}),
              ("craft", {"recipe": "plank"}, {"ok": True, "made": {"plank": 2}}),
              ("place", {"item": "plank"}, {"ok": True}),
              ("eat", {"item": "berries"}, {"ok": True}),
              ("dig", {"dir": "E", "with": "hoe"}, {"ok": True}),
              ("attack", {"target": 7, "with": "spear"}, {"ok": True, "killed": True}),
              ("craft", {"recipe": "axe"}, {"ok": False, "status": "interrupted"}),
              ("wear", {"item": "bandana"}, {"ok": False})]
    for tool, args, res in script:
        wire = ({"ok": True, "result": {k: v for k, v in res.items() if k != "ok"}} if res["ok"]
                else {"ok": False, "error": "interrupted" if res.get("status") else "bad_target"})
        a = sim.result(tool, args, res, "tusker", 8)
        b = live.result(tool, args, from_wire(wire), "tusker", 8)
        assert a == b
    sim.gained({}, {"wood": 2, "plank": 4}, {"plank": 4})
    live.gained({}, {"wood": 2, "plank": 4}, {"plank": 4})
    assert sim.done == live.done == {"craft_plank", "place_plank", "eat_berries", "till", "fought_armed", "defeat_tusker",
                                     "win_hurt", "collect_wood"}
    assert sim.counts == live.counts


def test_goals_pay_once_when_done():
    from nexel_agent.rl.goals import Goal
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    env.reset(5)
    env.goals.goal = Goal("craft", "plank", None, env.sim.tick, until=10 ** 9)
    env.a.inventory["wood"] = 3
    _, r, *_ = env.step(idx("craft_plank"))
    assert r >= 2.0 and env.goals.counts["craft"][0] == 1 and env.goals.goal.kind == "free"   # (achievement + goal)
    _, r, *_ = env.step(idx("craft_plank"))
    assert r < 1.0                                                                   # (nothing more for the same)


def test_a_goal_about_a_person_it_struck_is_not_paid():
    from nexel_agent.rl.goals import Goal
    for strike, paid in ((False, 1), (True, 0)):
        env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, shadow_pct=0)
        env.reset(5)
        f = neighbour_beside(env)
        env.goals.goal = Goal("avoid", None, f.a.name, env.sim.tick, until=10 ** 9)
        if strike:
            env.step(idx("attack"))
            env.step(0)
        f.a.x, f.a.y = env.a.x + 40, env.a.y + 40             # (far off for a minute: the goal's own measure of done)
        f.act = 10 ** 9
        for _ in range(140):
            env.step(0)
            f.a.x, f.a.y = env.a.x + 40, env.a.y + 40
        assert env.goals.counts["avoid"][0] == paid


def test_ask_goal_and_a_neighbour_who_says_yes(monkeypatch):
    monkeypatch.setattr(folk, "QUIET", 0)                       # (a neighbour who answers asks: a social one, #12)
    monkeypatch.setattr(folk, "ROUGH", 0)
    from nexel_agent.rl.goals import Goal
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    env.reset(5)
    sim, a, f = env.sim, env.a, env.folk[0]
    f.next = sim.tick + 1
    env.step(0)
    f.a.x, f.a.y, f.a.z = a.x + 1, a.y, 99
    sim.height(f.a)                                   # (on the ground now: the game's hearing counts height)
    f.rng.random = lambda: 0.0                        # (it says yes)
    f.a.inventory["wood"] = 2
    env.goals.goal = Goal("ask", "wood", f.a.name, sim.tick, until=10 ** 9)
    env.obs = sim.observe(a)
    assert env.encoded()["mask"][idx("ask")]
    env.step(idx("ask"))
    for _ in range(30):
        env.step(0)
        if env.goals.counts["ask"][0]:
            break
    assert env.goals.counts["ask"][0] == 1 and a.count("wood") >= 1



def test_the_goals_person_out_of_view_is_where_they_were_last_seen():
    """rl/actions.py person: in view as the game shows them ("seen" 1), kept on the goal; gone from view, where the
    player last saw them, from where it is now ("seen" 0), for the encoder, the mask and approach alike."""
    from nexel_agent.rl.actions import person
    from nexel_agent.rl.goals import Goal
    g = Goal("ask", "wood", "Mira")
    me = {"x": 10, "y": 10}
    assert person({"self": me, "entities": [{"kind": "agent", "name": "Mira", "dx": 3, "dy": -2, "hp": 20}]}, g)["seen"] == 1
    p = person({"self": {"x": 12, "y": 10}, "entities": []}, g)
    assert (p["dx"], p["dy"], p["seen"]) == (1, -2, 0) and g.seen == (13, 8, 20)
    assert person({"self": me, "entities": []}, Goal("ask", "wood", "Bo")) is None


def test_every_dig_swings_the_right_tool_from_the_pack_and_it_is_faster():
    """rl/actions.py tool_for: a dig takes the pack's tool the dug thing wants (the game's block table), dug down too;
    with no such tool, bare hands. In the sim a pickaxe takes the ground down sooner."""
    from nexel_agent.rl.actions import to_command
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0)
    times = {}
    for tool in (None, "pickaxe"):
        env.reset(3)
        if tool:
            env.a.inventory[tool] = 1
        obs = env.sim.observe(env.a)
        cmd = to_command(idx("dig_down"), obs, "S")
        assert cmd == ("dig_down", {"with": "pickaxe"} if tool else {})
        act, t0 = env.sim.submit(env.a, *cmd), env.sim.tick
        while act.result is None:
            env.sim.step()
        assert act.result["ok"]
        times[tool] = env.sim.tick - t0
    assert times["pickaxe"] < times[None]


def test_the_body_reads_the_game_with_the_vocabulary_it_learned():
    """A creature, an item or a recipe new in the game shifts no id the body learned (rl/vocab.py): the learned
    actions keep their rows, what is new comes after them, a char the body never learned reads as unknown."""
    from nexel_agent.rl import vocab
    from nexel_agent.rl.encode import CHAR_ID, ITEMS, KINDS
    assert ACTIONS[:len(vocab.ACTIONS)] == vocab.ACTIONS
    assert ITEMS == vocab.ITEMS and KINDS == vocab.KINDS and len(vocab.CHARS) + 1 == N_CHARS
    assert CHAR_ID["@"] == vocab.CHARS.index("@") + 1


def test_the_ores_and_tool_tiers_have_their_own_block_at_the_end():
    """(#24) What r57 learned keeps its place: its 200 actions (the game at 31963d92) and its 669 inputs. The pack's
    items that came after ITEMS (ores, ingots, tool tiers, the hob, the Eclipse Core) are a block after them; an action
    the game adds later goes after that, so a checkpoint only ever grows at the end (rl/train.py init_from)."""
    from nexel_agent.rl import vocab
    from nexel_agent.rl.encode import ACTIONS_V4, ITEMS_V2, V4, encode
    assert len(vocab.ACTIONS) == ACTIONS_V4 == 200 and V4 == 669 and V4 + len(ITEMS_V2) + N_ACTIONS - ACTIONS_V4 == VEC
    assert vocab.ACTIONS[-1] == ("build", None, None) and not set(ITEMS_V2) & set(vocab.ITEMS)
    o = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0).reset(5)
    obs = {"tick": 0, "grid": ["." * SIDE] * SIDE, "self": {"hp": 20, "hunger": 20, "inventory": {"iron_ore": 3, "wood": 1}}}
    v = encode(obs, last_action=ACTIONS_V4 - 1)["vec"]
    assert v[V4 + ITEMS_V2.index("iron_ore")] > 0 and np.count_nonzero(v[V4:V4 + len(ITEMS_V2)]) == 1
    assert v[V4 - 1] == 1 and o["vec"].shape == (VEC,)                       # (build, the last action r57 knew: the last slot before the block)


def test_the_home_is_dug_by_the_builder_only():
    """Live trial 5: between build goals the policy and its gathering dug 463 times on the home's site, its walls taken
    back up and a shaft 9 deep under it. On the home's tiles no dig is legal and no gathering digs; off them, as ever."""
    from nexel_agent.rl.actions import pursued
    from nexel_agent.rl.goals import Goal
    from nexel_agent.rl.social import SocialMemory
    env = SimEnv(ep_ticks=10 ** 6, roam=0, rivals=0, folk=0)
    env.reset(5)
    a = env.a
    env.sim.set_tile(a.x + 1, a.y, block="tree")
    social, wood = SocialMemory("Rook", ("wood",)), Goal("gather", "wood")
    obs = env.sim.observe(a)
    m = legal(obs, "E", social)
    assert m[idx("dig_E")] and m[idx("dig_down")] and pursued(obs, wood, social) == ("dig", {"dir": "E"})
    social.home = (a.x, a.y)                                       # (the hut's tiles, from here)
    m = legal(obs, "E", social)
    assert not m[idx("dig_E")] and not m[idx("dig_down")] and not m[idx("dig_E_axe")]
    assert pursued(obs, wood, social) != ("dig", {"dir": "E"})
    social.home = (a.x - 20, a.y)                                  # (a home elsewhere: here, as ever)
    assert legal(obs, "E", social)[idx("dig_E")]
