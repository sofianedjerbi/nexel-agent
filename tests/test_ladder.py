"""The computed progress block (ladder.py) and the failure feedback (fixes.py) the real-time player reads."""

import asyncio

import httpx

from megastructure.fastprompt import compact_obs, stable_prefix
from megastructure.fixes import failure_line
from megastructure.ladder import Record, progress_lines
from megastructure.plans import parse_command
from test_thinker_realtime import FakeGame, fake_obs, model_transport, thinker


def world(inv, extra=()):
    """A view with a tree at (0,-2), a rock at (4,2) and whatever `extra` puts at (x, y)."""
    o = fake_obs((0, 0))
    g = [list(r) for r in o["grid"]]
    for ch, x, y in (("M", 4, 2), *extra):
        g[10 + y][10 + x] = ch
    o["grid"] = ["".join(r) for r in g]
    o["self"]["inventory"] = dict(inv)
    return o


def test_a_station_recipe_without_a_workbench_says_craft_it_place_it_first():
    lines = progress_lines(world({"wood": 12}), "craft hoe", Record())
    step = next(x for x in lines if x.startswith("Your goal 'craft hoe'"))
    assert "1) craft workbench > 2) place N workbench > 3) get 1 stone: rock at (4,2): move_to 3 2, dig E > 4) craft hoe" in step


def test_a_carried_workbench_is_placed_and_a_placed_one_is_used():
    o = world({"wood": 6, "stone": 2, "workbench": 1})
    assert "next 1) place N workbench > 2) craft hoe" in " ".join(progress_lines(o, "make a hoe", None))
    o = world({"wood": 6, "stone": 2}, [("&", 1, 1)])
    assert "next 1) craft hoe." in " ".join(progress_lines(o, "make a hoe", None))
    o = world({"wood": 6, "stone": 2}, [("&", 4, -1)])                 # in view, not within 2 tiles
    assert "go within 2 tiles of the workbench at (4,-1)" in " ".join(progress_lines(o, "make a hoe", None))


def test_plenty_of_wood_turns_to_the_next_milestone_and_says_stop():
    rec = Record()
    rec.seen({"wood": 40})
    lines = progress_lines(world({"wood": 40}), "harvest wood", rec)
    assert lines[0] == "Milestones done: 1; next: workbench placed."
    assert lines[1].startswith("Next milestone 'workbench placed': next 1) craft workbench > 2) place N workbench")
    assert "Plenty of wood 40: stop gathering that" in lines[2]


def test_milestones_fall_in_order_and_persist(tmp_path):
    rec = Record(tmp_path / "progress.json")
    rec.seen({"wood": 9}, 10.0)
    rec.did("place", {"dir": "S", "item": "workbench"}, True, 20.0)
    rec.did("place", {"dir": "S", "item": "wood"}, False, 21.0)       # a failure is no progress
    rec.seen({"stone": 3, "fiber": 1, "axe": 1, "pickaxe": 1, "hoe": 1}, 30.0)
    again = Record(tmp_path / "progress.json")
    assert again.reached == {"wood": 10.0, "workbench placed": 20.0, "stone + fiber": 30.0, "axe, pickaxe, hoe": 30.0}
    assert again.next_milestone() == "campfire" and "wood" not in again.placed


def test_failures_say_what_would_work():
    o = world({"wood": 20, "fence": 1})
    say = lambda t, a, e, d="": failure_line(t, a, {"ok": False, "error": e, "detail": d}, o)
    line = say("place", {"dir": "N", "item": "wood"}, "not_placeable")
    assert line.startswith("place N wood FAILED not_placeable: wood is a material") and "you can place: fence" in line
    assert "workbench" in line
    assert "first craft workbench > place N workbench" in say("craft", {"recipe": "hoe"}, "need_station")
    assert "a hoe: craft workbench > place N workbench" in say("dig", {"dir": "S"}, "need_tool", "tilling needs a hoe in hand")
    assert "2 wood (have 20), 3 stone (have 0), 1 fiber (have 0)" in say("craft", {"recipe": "pickaxe"}, "missing_items")
    assert "free flat sides now: N, E, S, W" in say("place", {"dir": "N", "item": "fence"}, "cliff_edge")
    assert "pick_up only takes items lying on your own tile" in say("pick_up", {}, "nothing_here")
    o["self"]["inventory"]["hoe"] = 1
    assert "To till: dig S hoe" in say("dig", {"dir": "S"}, "need_tool", "tilling needs a hoe in hand")


def test_a_hill_edge_is_no_free_side():
    o = world({"workbench": 1})
    o["heights"] = ["0" * 21] * 9 + ["1" * 21] * 12          # lower ground from y=-2 up: N of you is the hill's edge
    assert "next 1) place E workbench." in " ".join(progress_lines(o, "place workbench", None))


def test_place_forms_the_model_writes_all_parse():
    assert parse_command("place workbench S") == parse_command("place S workbench") == ("place", {"dir": "S", "item": "workbench"})
    assert parse_command("place dir=S item=workbench") == ("place", {"dir": "S", "item": "workbench"})
    assert parse_command("write_sign N go N now") == ("write_sign", {"dir": "N", "text": "go N now"})


def test_prompt_stays_short_with_the_progress_block():
    prefix = stable_prefix("gatsu0", "A grim swordsman.")
    assert "Milestones: 1 wood > 2 workbench placed" in prefix and len(prefix) < 3600
    user = compact_obs(world({"wood": 45}), goal="craft hoe", record=Record())
    assert "Your goal 'craft hoe': next 1) craft workbench" in user and len(user) < 1700


def test_the_next_plan_reads_the_fix_for_a_failed_place(tmp_path):
    plans = ['{"goal":"build","do":["place N wood"]}', '{"goal":"build","do":["dig N"]}']
    transport, calls = model_transport(plans, delay=0.2)
    t, game = thinker(tmp_path, chat=False), FakeGame(fail={"place N"})
    game.inv = {"wood": 20}

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=2.0 / 60)
    asyncio.run(go())
    later = [c[1]["messages"][1]["content"] for c in calls if c[1].get("stream")][1:]
    assert any("place N wood FAILED" in u and "Milestones done:" in u for u in later)
    assert (tmp_path / "Vex" / "progress.json").exists()


def test_digging_bare_ground_says_so_and_look_far_says_what_is_out_there():
    from megastructure.fixes import look_far_line
    o = world({})
    line = failure_line("dig", {"dir": "S"}, {"ok": False, "error": "need_tool", "detail": "tilling needs a hoe in hand"}, o)
    assert "that is bare ground: digging it tills it" in line and "dig toward a tree, rock or bush" in line
    assert "attack takes a being's id" in failure_line("attack", {"target": 30}, {"ok": False, "error": "no_target"}, o)
    res = {"rows_near_to_far": ["....", "..T.", "M...", "~~~~"], "agents": [{"name": "Ana", "dx": 3, "dy": -20}]}
    assert look_far_line({"dir": "N"}, res) == "look_far N -> tree ~13 tiles, rock ~15 tiles; water from ~17 tiles; people: Ana (+3,-20) (N of you)"
    assert look_far_line({"dir": "E"}, {"rows_near_to_far": ["..b."]}) == "look_far E -> berry_bush ~11 tiles; no tree or rock (E of you)"
