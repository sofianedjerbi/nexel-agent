"""The facts the real-time player reads about its goal (ladder.py) and about a failure (fixes.py): what happened, what
a recipe takes, where its parts are. Never a step to take ("no do X when Y": owner, 2026-11-23)."""

import asyncio

import httpx

from megastructure.fastprompt import compact_obs, stable_prefix
from megastructure.fixes import failure_line, look_far_line
from megastructure.ladder import Record, free_dirs, progress_lines
from megastructure.plans import parse_command
from test_thinker_realtime import FakeGame, fake_obs, model_transport, thinker

ORDERS = ("first ", "then ", "next ", "stop ", "should", "try ", "make something", "explore")


def world(inv, extra=()):
    """A view with a tree at (0,-2), a rock at (4,2) and whatever `extra` puts at (x, y)."""
    o = fake_obs((0, 0))
    g = [list(r) for r in o["grid"]]
    for ch, x, y in (("M", 4, 2), *extra):
        g[10 + y][10 + x] = ch
    o["grid"] = ["".join(r) for r in g]
    o["self"]["inventory"] = dict(inv)
    return o


def no_orders(text):
    assert not any(w in text.lower() for w in ORDERS), text


def test_a_goal_gets_its_recipe_what_you_have_and_where_the_rest_is():
    line = " ".join(progress_lines(world({"wood": 12}), "craft hoe", Record()))
    assert "hoe = 2 wood (have 12), 1 stone (have 0) @workbench" in line
    assert "a placed workbench within 2 tiles: none in view; you carry none; workbench = 4 wood" in line
    assert "stone comes from rock or ore or cave wall; nearest: rock at (4,2): stand at (3,2), it is E of there" in line
    no_orders(line)


def test_the_workbench_fact_follows_what_is_there():
    assert "you carry 1" in " ".join(progress_lines(world({"wood": 6, "stone": 2, "workbench": 1}), "make a hoe", None))
    assert "within 2 tiles: yes" in " ".join(progress_lines(world({"wood": 6, "stone": 2}, [("&", 1, 1)]), "make a hoe", None))
    far = " ".join(progress_lines(world({"wood": 6, "stone": 2}, [("&", 4, -1)]), "make a hoe", None))
    assert "within 2 tiles: no; one is at (4,-1), 5 tiles away" in far


def test_a_material_goal_and_the_milestones_are_facts():
    rec = Record()
    rec.seen({"wood": 40})
    lines = progress_lines(world({"wood": 40}), "harvest wood", rec)
    assert lines[0] == "Milestones reached: 1."
    assert lines[1].startswith("For your goal: wood comes from tree") and "nearest: tree at (0,-2)" in lines[1]
    no_orders(" ".join(lines))


def test_milestones_fall_in_order_and_persist(tmp_path):
    rec = Record(tmp_path / "progress.json")
    rec.seen({"wood": 9}, 10.0)
    rec.did("place", {"dir": "S", "item": "workbench"}, True, 20.0)
    rec.did("place", {"dir": "S", "item": "wood"}, False, 21.0)       # a failure is no progress
    rec.seen({"stone": 3, "fiber": 1, "axe": 1, "pickaxe": 1, "hoe": 1}, 30.0)
    again = Record(tmp_path / "progress.json")
    assert again.reached == {"wood": 10.0, "workbench placed": 20.0, "stone + fiber": 30.0, "axe, pickaxe, hoe": 30.0}
    assert again.next_milestone() == "campfire" and "wood" not in again.placed


def test_failures_say_what_happened_and_the_rule():
    o = world({"wood": 20, "fence": 1})
    say = lambda t, a, e, d="": failure_line(t, a, {"ok": False, "error": e, "detail": d}, o)
    line = say("place", {"dir": "N", "item": "wood"}, "not_placeable")
    assert line.startswith("place N wood FAILED not_placeable: wood is a material and cannot be placed")
    assert "placeable things you carry: fence" in line and "placeable things made from wood: workbench" in line
    station = say("craft", {"recipe": "hoe"}, "need_station")
    assert "hoe = 2 wood (have 20), 1 stone (have 0) @workbench; a placed workbench within 2 tiles: none in view" in station
    bare = say("dig", {"dir": "S"}, "need_tool", "tilling needs a hoe in hand")
    assert "S of you is bare ground; digging bare ground tills it, which takes a hoe in hand" in bare
    assert "Nearest thing a dig gathers: tree at (0,-2): stand at (0,-1), it is N of there." in bare
    assert "2 wood (have 20), 3 stone (have 0), 1 fiber (have 0)" in say("craft", {"recipe": "pickaxe"}, "missing_items")
    assert "free flat sides next to you now: N, E, S, W" in say("place", {"dir": "N", "item": "fence"}, "cliff_edge", "edge")
    assert "pick_up takes items lying on your own tile; none lie there" in say("pick_up", {}, "nothing_here")
    assert "attack takes a being's id" in say("attack", {"target": 30}, "no_target")
    for x in (line, station, bare):
        no_orders(x)


def test_a_hill_edge_is_no_free_side():
    o = world({"workbench": 1})
    o["heights"] = ["0" * 21] * 9 + ["1" * 21] * 12          # lower ground from y=-2 up: N of you is the hill's edge
    assert free_dirs(o) == ["E", "S", "W"]


def test_place_forms_the_model_writes_all_parse():
    assert parse_command("place workbench S") == parse_command("place S workbench") == ("place", {"dir": "S", "item": "workbench"})
    assert parse_command("place dir=S item=workbench") == ("place", {"dir": "S", "item": "workbench"})
    assert parse_command("write_sign N go N now") == ("write_sign", {"dir": "N", "text": "go N now"})
    assert parse_command("dig_at tree with axe") == ("dig_at", {"text": "tree with axe"})


def test_prompt_states_facts_and_stays_short():
    prefix = stable_prefix("gatsu0", "A grim swordsman.")
    assert "Milestones: 1 wood > 2 workbench placed" in prefix and "done_when" in prefix and len(prefix) < 4000
    user = compact_obs(world({"wood": 45}), goal="craft hoe", record=Record())
    assert "For your goal: hoe = 2 wood (have 45)" in user and len(user) < 1700
    assert "tree (0,-2): stand at (0,-1), it is N" in user


def test_the_next_plan_reads_what_failed(tmp_path):
    plans = ['{"goal":"build","do":["place N wood"]}', '{"goal":"build","do":["dig N"]}']
    transport, calls = model_transport(plans, delay=0.2)
    t, game = thinker(tmp_path, chat=False), FakeGame(fail={"place N"})
    game.inv = {"wood": 20}

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=2.0 / 60)
    asyncio.run(go())
    later = [c[1]["messages"][1]["content"] for c in calls if c[1].get("stream")][1:]
    assert any("place N wood FAILED" in u and "Milestones reached:" in u for u in later)
    assert (tmp_path / "Vex" / "progress.json").exists()


def test_look_far_says_what_is_out_there():
    res = {"rows_near_to_far": ["....", "..T.", "M...", "~~~~"], "agents": [{"name": "Ana", "dx": 3, "dy": -20}]}
    assert look_far_line({"dir": "N"}, res) == "look_far N -> tree ~13 tiles, rock ~15 tiles; water from ~17 tiles; people: Ana (+3,-20) (N of you)"
    seen = {}
    line = look_far_line({"dir": "N"}, res, {"x": 0, "y": 0}, seen)
    assert "tree ~13 tiles at (-4,-13)" in line and seen["tree"] == (-4, -13)
    o = world({})
    o["_seen"] = seen
    far = failure_line("dig_at", {"text": "tree"}, {"ok": False, "error": "not_found", "detail": "no 'tree' in view"}, o)
    assert "you saw one far off with look_far at (-4,-13)" in far
    assert look_far_line({"dir": "E"}, {"rows_near_to_far": ["..b."]}) == "look_far E -> berry_bush ~11 tiles; no tree or rock (E of you)"


def test_stone_only_from_ore_says_the_ore_needs_a_pickaxe():
    o = fake_obs((0, 0))
    g = [list(r) for r in o["grid"]]
    g[10][11] = "*"                                                      # ore E of you, no rock anywhere
    o["grid"] = ["".join(r) for r in g]
    o["self"]["inventory"] = {"wood": 10, "fiber": 2}
    line = " ".join(progress_lines(o, "craft pickaxe", None))
    assert "nearest: ore at (1,0) (needs a pickaxe in hand: you carry none): next to you, E" in line


def test_what_was_seen_before_is_remembered_where_it_was(tmp_path):
    t = thinker(tmp_path)
    t.plan_messages(world({}))                                           # a tree at (0,-2), a rock at (4,2)
    t.goal = "craft workbench"
    user = t.plan_messages(fake_obs((30, 0), tree=False))[1]["content"]
    assert "wood comes from tree or stump or bare fruit tree: none in view; last seen: tree at (0,-2)" in user
