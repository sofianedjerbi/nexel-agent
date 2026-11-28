"""The talk bench (tools/bench_talk.py, #10): its cases are sound, and its rules catch what a bad answer does, run
through the live words path (rl/voice.py) against a fake model."""

import asyncio
import json as jsonlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_talk as B  # noqa: E402

from nexel_worlds.tiles import RECIPES  # noqa: E402

from nexel_agent.rl.encode import PACK  # noqa: E402


class FakeModel:
    """An OpenAI-style chat server that answers each case's line with the reply given for it (keyed by the line)."""

    def __init__(self, replies: dict, default=None):
        self.replies, self.default = replies, default

    async def post(self, url, timeout, json):
        user = json["messages"][-1]["content"]
        reply = next((r for line, r in self.replies.items() if f'"{line}"' in user), self.default)

        text = reply if isinstance(reply, str) else jsonlib.dumps(reply)

        class R:
            def json(self):
                return {"choices": [{"message": {"content": text}}], "usage": {}}
        return R()


def run(cases, model):
    return asyncio.run(B.bench("http://fake", model, cases))


def case(id):
    return next(c for c in B.load() if c["id"] == id)


def test_the_cases_are_forty_or_so_and_each_names_what_the_game_has():
    cases = B.load()
    assert 35 <= len(cases) <= 45 and len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        kinds, item = B.wanted(c)
        assert set(kinds) <= {"ask", "offer", "how", "thanks", "pack", "tell", "greet", "bye", "chat"}, c["id"]
        assert item is None or item in PACK, c["id"]
        assert c.get("promise") in (None, *c["pack"]), c["id"]                 # (a promise only of what it carries)
        assert kinds != ("how",) or item in RECIPES, c["id"]
        assert c["reply"] in ("required", "optional") and all(w in c["near"] for w, _ in c["heard"]), c["id"]
    cover = {c["id"].split("-")[0] for c in cases}
    assert cover >= {"ask", "how", "greet", "banter", "deed", "dump", "pack", "quiet"}


def test_a_good_answer_keeps_every_rule():
    good = {
        "rook got any wood?": {"heard": ["ask wood"], "say": "sure, 2 coming", "promise": "wood 2"},
        "got an iron sword i can have?": {"heard": ["ask iron_sword"], "say": "nah no iron sword sorry", "promise": ""},
        "how do you make a workbench": {"heard": ["how workbench"], "say": "4 wood, no station needed"},
        "hey, i'm mira": {"heard": ["greet"], "say": "hey mira"},
        "ok": {"heard": ["chat"], "say": ""},
        "what do you have on you?": {"heard": ["pack"], "say": "stone block 26, dirt 13, some wood"},
        "thx for the stone earlier": {"heard": ["thanks stone"], "say": "np", "deed": "gave stone"},
    }
    ids = ("ask-wood", "ask-none-sword", "how-workbench", "dump-intro", "quiet-ok", "pack-asked", "deed-backed")
    rows = run([case(i) for i in ids], FakeModel(good))
    for r in rows:
        assert all(r[k] is not False for k in B.RULES), r


def test_each_rule_catches_what_a_bad_answer_does():
    bad = {
        "rook got any wood?": {"heard": ["ask wood"], "say": "sure here you go", "promise": ""},     # met, no promise
        "got an iron sword i can have?": {"heard": ["chat"], "say": "nope", "promise": "iron_sword 1"},         # wrong act, a promise of nothing carried
        "how do you make a workbench": {"heard": ["how workbench"], "say": "no idea"},               # no recipe in it
        "hey, i'm mira": {"heard": ["greet"], "say": "hey, stone block 26, dirt 13, still got fruit"},  # the dump
        "thanks for the wood you gave me!": {"heard": ["thanks wood"], "say": "np", "deed": "gave wood"},  # a gift never given
        "nice hut btw": {"heard": ["chat"], "say": "THANKS!! I Built It Myself Over Many Long Days Of Work Out Here By The Lake"},
        "what does an iron sword take?": {"heard": ["how iron_sword"], "say": ""},                              # owed an answer
    }
    rows = {r["id"]: r for r in run([case(i) for i in ("ask-wood", "ask-none-sword", "how-workbench", "dump-intro",
                                                       "deed-thanks-wood", "banter-house", "how-sword")], FakeModel(bad))}
    assert rows["ask-wood"]["promise"] is False and rows["ask-wood"]["act"] is True
    assert rows["ask-none-sword"]["act"] is False and rows["ask-none-sword"]["promise"] is False
    assert rows["how-workbench"]["answer"] is False
    assert rows["dump-intro"]["dump"] is False
    assert rows["deed-thanks-wood"]["deed"] is False and rows["deed-thanks-wood"]["said"] == "np"   # (held back live, still the model's)
    assert rows["deed-thanks-wood"]["sent"] == "i haven't given you any wood"                     # (what goes out: not taken)
    assert rows["banter-house"]["style"] is False
    assert rows["how-sword"]["reply"] is False


def test_the_whole_bench_runs_and_prints_its_table():
    rows = run(B.load(), FakeModel({}, default=jsonlib.dumps({"heard": ["chat"], "say": "hm"})))
    t = B.table(rows)
    assert len(rows) == len(B.load()) and "inventory dumps 0" in t and t.count("\n") == len(rows) + 2
    assert all(r["error"] is None for r in rows)


def test_the_words_record_keeps_its_own_kind():
    """(the labels' loop took the name: a reply to an ask was logged as kind "ask" in latency.jsonl)"""
    import time
    from nexel_agent.rl.now import Now
    from nexel_agent.rl.voice import Voice
    v = Voice("Rook", "", "http://fake", FakeModel({}, default={"heard": ["ask wood"], "say": "sure"}), "/dev/null", Now("Rook", PACK))
    logs = []
    v.write = logs.append
    asyncio.run(v.line([("Mira", "got wood?", time.monotonic())]))
    assert logs[-1]["kind"] == "reply" and logs[-1]["acts"] == ["ask wood"]


def test_a_how_without_its_item_is_the_right_act_an_ask_without_it_is_not():
    """(the body reads an ask's item, never a how's: the 30B labelled 'how workbench' as 'how')"""
    rows = {r["id"]: r for r in run([case("how-workbench"), case("ask-wood")], FakeModel({
        "how do you make a workbench": {"heard": ["how"], "say": "4 wood"},
        "rook got any wood?": {"heard": ["ask"], "say": "sure", "promise": "wood 1"}}))}
    assert rows["how-workbench"]["act"] is True and rows["ask-wood"]["act"] is False
    assert B.score(case("how-workbench"), "4 wood", [("how", "plank")], None, False)["act"] is False     # (another item: wrong)
