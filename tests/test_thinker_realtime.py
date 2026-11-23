"""The real-time thinker: plan parsing and streaming, running a plan and dropping the rest on a failure,
pipelining (the next model call runs while the plan is still being played; nothing scripted meanwhile), chat fast path.
No model and no game server: a fake game and an httpx mock transport."""

import asyncio
import json
import time

import httpx

from megastructure.fastprompt import compact_obs, stable_prefix
from megastructure.planner import Item
from megastructure.plans import PlanStream, command_text, parse_command, parse_plan
from megastructure.thinker import Thinker


# ------------------------------------------------------------------ parsing
def test_commands_parse_positional_and_key_value():
    assert parse_command("dig N axe") == ("dig", {"dir": "N", "with": "axe"})
    assert parse_command("dig N with pickaxe") == ("dig", {"dir": "N", "with": "pickaxe"})
    assert parse_command("move_to 3 -4") == ("move_to", {"x": 3, "y": -4})
    assert parse_command("place E workbench") == ("place", {"dir": "E", "item": "workbench"})
    assert parse_command("give #7 wood 2") == ("give", {"target": 7, "item": "wood", "n": 2})
    assert parse_command("dig dir=w with=axe") == ("dig", {"dir": "W", "with": "axe"})
    assert parse_command("say hi, want to trade with me?") == ("say", {"text": "hi, want to trade with me?"})
    assert parse_command({"tool": "craft", "args": {"recipe": "plank"}}) == ("craft", {"recipe": "plank"})
    for bad in ("dig axe", "move_to here", "fly N", "", 7, "walk"):
        assert parse_command(bad) is None, bad


def test_plan_parses_and_drops_bad_commands():
    cmds, goal, note = parse_plan('ok {"goal":"get wood","do":["move_to 4 -2","dance","dig N"],"note":"trees east"}')
    assert cmds == [("move_to", {"x": 4, "y": -2}), ("dig", {"dir": "N"})]
    assert (goal, note) == ("get wood", "trees east")
    assert parse_plan("no json here")[0] == []


def test_stream_hands_out_each_command_as_its_string_closes():
    text = '{"goal":"wood","do":["move_to 1 2", "dig N","craft plank"]}'
    ps, got = PlanStream(), []
    for i, ch in enumerate(text):
        for c in ps.feed(ch):
            got.append((i, c))
    assert [c for _, c in got] == ["move_to 1 2", "dig N", "craft plank"]
    assert got[0][0] < text.index("dig N")              # the first one is out before the second is even written
    assert ps.goal == "wood"


def test_prompt_is_short_and_its_prefix_stable():
    assert stable_prefix("Vex", "a trader") == stable_prefix("Vex", "a trader")
    user = compact_obs(fake_obs((0, 0)))
    assert len(stable_prefix("Vex", "a trader")) < 4000 and len(user) < 1500
    assert "tree (0,-2): stand at (0,-1), it is N" in user


# ---------------------------------------------------------------- fakes
def fake_obs(pos, events=(), tree=True):
    grid = [["."] * 21 for _ in range(21)]
    r, c = 10 + (-2 - pos[1]), 10 - pos[0]               # a tree at (0,-2)
    if tree and 0 <= r < 21 and 0 <= c < 21:
        grid[r][c] = "T"
    grid[10][10] = "@"
    return {"tick": 1, "time": "day 1, 09:00", "grid": ["".join(r) for r in grid], "legend": {".": "grass", "T": "tree"},
            "self": {"id": 1, "name": "Vex", "x": pos[0], "y": pos[1], "hp": 20, "hunger": 20, "inventory": {},
                     "standing_on": "grass", "sheltered": False, "z": 0},
            "entities": [], "items": [], "signs": [], "events": list(events)}


class FakeGame:
    def __init__(self, fail=(), act_s=0.05, events=None):
        self.pos, self.fail, self.act_s = [0, 0], set(fail), act_s
        self.acts: list[tuple[float, str, dict]] = []
        self.events = list(events or [])

    async def observe(self):
        ev, self.events = self.events, []
        obs = fake_obs(tuple(self.pos), ev, tree=getattr(self, "tree", True))
        obs["self"]["inventory"] = dict(getattr(self, "inv", {}))
        obs["self"]["hunger"] = getattr(self, "hunger", 20)
        return obs

    async def act(self, tool, **args):
        self.acts.append((time.monotonic(), tool, args))
        await asyncio.sleep(self.act_s)
        if f"{tool} {args.get('dir', '')}".strip() in self.fail:
            return {"ok": False, "error": "nothing_there", "detail": "x"}
        if tool == "move_to":
            self.pos = [args["x"], args["y"]]
        return {"ok": True, "result": {"ticks": 1, "changed": {}}}


def model_transport(plans, delay=0.3, chat_line="Sure, take two.", calls=None):
    """Streams plans[i] for the i-th plan call (slowly), answers a chat call at once."""
    calls = calls if calls is not None else []

    async def handler(request):
        body = json.loads(request.content)
        calls.append((time.monotonic(), body))
        if not body.get("stream"):
            return httpx.Response(200, json={"choices": [{"message": {"content": chat_line}}], "usage": {}})
        text = plans[min(len([c for c in calls if c[1].get("stream")]) - 1, len(plans) - 1)]

        async def chunks():
            n = 6
            for i in range(0, len(text), n):
                await asyncio.sleep(delay * n / len(text))
                yield ("data: " + json.dumps({"choices": [{"delta": {"content": text[i:i + n]}}]}) + "\n\n").encode()
            yield b'data: {"choices": [], "usage": {"prompt_tokens": 900, "prompt_tokens_details": {"cached_tokens": 800}}}\n\n'
            yield b"data: [DONE]\n\n"
        return httpx.Response(200, content=chunks(), headers={"content-type": "text/event-stream"})
    return httpx.MockTransport(handler), calls


def thinker(tmp_path, cards=False, **kw):
    """A thinker on the fake model; cards=False: no card-writing calls (tests that count the model's calls)."""
    t = Thinker("Vex", "a trader", data_dir=tmp_path, model_url="http://model/v1/chat/completions", **kw)
    if not cards:
        t.ask_cards = lambda *a, **k: None
    return t


# ---------------------------------------------------------------- body
def step(t, game, n):
    async def go():
        for _ in range(n):
            obs = await t.observe(game)
            t.new_plan.clear()
            it = t.next_item(obs)
            if it is None:
                return
            await t.execute(game, it, obs)
    asyncio.run(go())


def test_plan_runs_in_order_and_a_failure_drops_the_rest(tmp_path):
    t, game = thinker(tmp_path), FakeGame(fail={"dig N"})
    t.queue.extend(Item(tool, args, "model", "wood", None, t.epoch) for tool, args in
                   [("move_to", {"x": 0, "y": -1}), ("dig", {"dir": "N"}), ("craft", {"recipe": "plank"})])
    step(t, game, 3)
    tools = [a[1] for a in game.acts]
    assert tools == ["move_to", "dig"]                                  # the third was dropped, nothing else done
    assert t.stats["aborts"] == 1
    assert any("dig N FAILED nothing_there" in h for h in t.history)    # the failure, said back as it was


def test_a_relative_command_from_elsewhere_is_not_sent_and_the_model_is_told(tmp_path):
    t, game = thinker(tmp_path), FakeGame()
    game.pos = [3, 0]                                                    # the body is not where the plan was made
    t.queue.append(Item("dig", {"dir": "N"}, "model", "wood", (0, -1), t.epoch))
    step(t, game, 2)
    assert game.acts == [] and any("was planned from (0, -1)" in h for h in t.history)


def test_nothing_planned_the_body_waits_for_the_model(tmp_path):
    """No scripted filler: before the first plan, and between plans, the body does nothing of its own."""
    plan = '{"goal":"get wood","do":["move_to 0 -1","dig N","dig N","dig N"]}'
    transport, calls = model_transport([plan], delay=0.4)
    t, game = thinker(tmp_path, chat=False), FakeGame(act_s=0.35)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await t.play(game, http, minutes=3.5 / 60)
    s = asyncio.run(go())
    plan_calls = [c[0] for c in calls if c[1].get("stream")]
    model_acts = [a for a in game.acts if a[1] == "dig"]
    assert len(plan_calls) >= 2
    assert game.acts[0][0] > plan_calls[0]                               # nothing before the model spoke
    assert all(k.split(":")[0] in ("model", "chat") for k in t.stats if ":" in k)
    assert any(model_acts[0][0] < c < model_acts[-1][0] for c in plan_calls[1:])   # a call overlapped the plan
    assert s["model_calls"] >= 2
    assert calls[0][1]["messages"][0]["content"] == calls[-1][1]["messages"][0]["content"]   # cacheable prefix


def test_stale_plan_is_dropped_after_an_abort(tmp_path):
    t = thinker(tmp_path, chat=False)
    t.queue.append(Item("dig", {"dir": "N"}, "model", "", None, t.epoch))
    old = t.epoch
    t.abort("hit")
    assert t.epoch == old + 1 and not t.queue
    t.queue.append(Item("dig", {"dir": "N"}, "model", "", None, old))  # a command from a plan still streaming
    assert t.next_item(fake_obs((0, 0))) is None


def test_the_same_failed_command_is_not_resent_within_a_second(tmp_path):
    t, game = thinker(tmp_path), FakeGame(fail={"dig N"})
    for _ in range(2):
        t.queue.append(Item("dig", {"dir": "N"}, "model", "", None, t.epoch))
        step(t, game, 1)
    assert len(game.acts) == 1 and t.stats["backoff"] == 1


def test_remember_and_forget_are_the_models_own_commands(tmp_path):
    t, game = thinker(tmp_path), FakeGame()
    assert parse_command("remember Ana owes me 3 wood") == ("remember", {"text": "Ana owes me 3 wood"})
    t.queue.append(Item("remember", {"text": "home by the big rock"}, "model", "", None, t.epoch))
    step(t, game, 1)
    assert game.acts == [] and "[1] home by the big rock (0,0)" in t.mem.notes_text()
    t.queue.append(Item("forget", {"n": 1}, "model", "", None, t.epoch))
    step(t, game, 1)
    assert t.mem.notes_text() == ""


# ---------------------------------------------------------------- chat
def test_someone_talking_gets_an_answer_fast(tmp_path):
    plan = '{"goal":"get wood","do":["dig N","dig N","dig N","dig N"]}'
    transport, calls = model_transport([plan], delay=3.0)                # a slow plan call: chat must not wait for it
    game = FakeGame(act_s=0.1, events=[{"type": "say", "from": "Ana", "text": "hi Vex, any wood?"}])
    t = thinker(tmp_path)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            t0 = time.monotonic()
            await t.play(game, http, minutes=1.2 / 60)
            return t0
    t0 = asyncio.run(go())
    says = [a for a in game.acts if a[1] == "say"]
    assert says and says[0][2]["text"] == "Sure, take two." and says[0][0] - t0 < 1.0
    chat_call = next(c for c in calls if not c[1].get("stream"))
    assert chat_call[1]["max_tokens"] <= 80 and "Ana says: hi Vex" in chat_call[1]["messages"][1]["content"]


def test_goal_names_what_its_recipe_still_lacks():
    from megastructure.fastprompt import needs_for
    grid = ["." * 5] * 5
    line = needs_for("get pickaxe", {"wood": 14}, grid)
    assert "2 wood (have 14)" in line and "3 stone (have 0)" in line and "workbench within 2 tiles (none)" in line
    assert needs_for("get wood", {}, grid) == "" and needs_for("make torches", {}, grid).startswith("torch needs")


def test_attack_by_a_name_or_kind_in_view_means_its_id():
    from megastructure.planner import by_id
    obs = fake_obs((0, 0))
    obs["entities"] = [{"id": 12, "kind": "agent", "name": "Mira", "dx": 1, "dy": 0}, {"id": 30, "kind": "tusker", "dx": 3, "dy": 0}]
    assert parse_command(by_id("attack Mira with fists", obs)) == ("attack", {"target": 12, "with": "fists"})
    assert parse_command(by_id("attack tusker", obs)) == ("attack", {"target": 30})
    assert by_id("attack Bob", obs) == "attack Bob"
