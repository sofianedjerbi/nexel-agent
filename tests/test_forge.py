"""The skill forge with a fake model: the sandbox, the simulated worlds, the library. The reference skills here exist
only to prove the bench works (they are never shown to a model nor used by a player)."""

import json

import httpx

from megastructure.skills import forge as F
from megastructure.skills.library import Library
from megastructure.skills.sandbox import Sandbox, check
from megastructure.skills.simworld import SimBody

GET_WOOD = '''
def get_wood(n):
    """Dig trees until n wood."""
    tries = 0
    while have("wood") < n and tries < 12:
        tries += 1
        if not act("dig_at tree")["ok"]:
            for d in "NESW":
                act("look_far " + d)
    return have("wood") >= n
'''
MAKE_WORKBENCH = '''
def make_workbench():
    """Wood, craft a workbench, put it down."""
    if have("wood") < 4 and not get_wood(4):
        return False
    act("craft workbench")
    return act("place_near workbench")["ok"]
'''


def test_the_sandbox_refuses_what_is_not_allowed_and_reports_errors():
    assert "not allowed" in check("f", "import os\ndef f(): pass")
    assert "must define a function named f" in check("f", "def g(): pass")
    body = SimBody(1)
    run = Sandbox(body.do, body.observe).run("f", "def f():\n    return 1 / 0\n")
    assert run["error"].startswith("ZeroDivisionError") and "line 2" in run["error"]


def test_reference_skills_pass_in_a_simulated_world():
    body = SimBody(1)
    run = Sandbox(body.do, body.observe).run("make_workbench", MAKE_WORKBENCH, (), {"get_wood": GET_WOOD})
    assert run["error"] is None and run["value"] is True, run
    assert F.passes(body, {"placed": "workbench"})


def test_the_forge_keeps_a_skill_that_passes_and_retries_one_that_fails(tmp_path, monkeypatch):
    answers = iter(["```text\nnaive\n```\n```python\ndef get_wood(n):\n    \"\"\"x\"\"\"\n    return False\n```",
                    f"```text\nfixed: dig trees\n```\n```python{GET_WOOD}```"])
    asked = []

    def fake_ask(url, msgs, max_tokens=1800):
        asked.append(msgs)
        return next(answers), 1.0
    monkeypatch.setattr(F, "ask", fake_ask)
    monkeypatch.setattr(F, "SEEDS", (1, 7))
    monkeypatch.setattr(F, "PASS", 2)
    lib = Library(tmp_path / "_skills")
    rows = F.forge("http://model", lib, tries=3, only=["get_wood"])
    assert rows[0]["ok"] and [a["pass"] for a in rows[0]["attempts"]] == ["0/2", "2/2"]
    assert "It passed in 0 of 2 worlds" in asked[1][-1]["content"]
    again = Library(tmp_path / "_skills")
    assert "get_wood" in again.sources() and again.index["get_wood"]["sim"] == "2/2"
    assert again.lines().startswith("skill get_wood N: Dig trees until n wood. (sim 2/2, play 0/0)")
    json.loads((tmp_path / "_skills" / "forge_log.json").read_text())


def test_a_skill_that_keeps_losing_in_play_is_no_longer_offered(tmp_path):
    lib = Library(tmp_path)
    lib.add("get_wood", GET_WOOD, "n", {"have": {"wood": 6}}, "test", "5/5")
    for _ in range(10):
        lib.outcome("get_wood", False)
    assert lib.offered() == []
    assert httpx                                          # (the forge's real calls go through httpx)


def test_a_plan_runs_a_library_skill_to_its_end_and_its_test_says_won(tmp_path):
    import asyncio
    from test_thinker_realtime import FakeGame, model_transport, thinker
    lib = Library(tmp_path / "_skills")
    lib.add("scan", 'def scan():\n    """Look all four ways."""\n    for d in "NESW":\n        act("look_far " + d)\n    return True\n',
            "", {"hp_at_least": 1}, "test", "5/5")
    t, game = thinker(tmp_path, chat=False), FakeGame(act_s=0.05)
    transport, calls = model_transport(['{"goal":"look","do":["skill scan"]}'], delay=0.2)

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            await t.play(game, http, minutes=1.5 / 60)
    asyncio.run(go())
    looks = [a[2]["dir"] for a in game.acts if a[1] == "look_far"]
    assert looks[:4] == ["N", "E", "S", "W"]
    assert Library(tmp_path / "_skills").index["scan"]["wins"] >= 1
    assert "skill scan: Look all four ways." in calls[-1][1]["messages"][1]["content"]
