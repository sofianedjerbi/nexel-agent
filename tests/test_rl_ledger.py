"""The favour ledger (rl/now.py Now: trust, who owes whom, who had what), from outcomes only, and the planner's facts
from it (rl/mind.py lacks): what its aims lack and who had it. Facts, never a rule of what to do."""

import time

from nexel_agent.rl import now as N
from nexel_agent.rl.mind import Mind
from nexel_agent.rl.now import Now

from test_rl_mind import FakeHTTP


def obs(events=(), entities=(), inv=None):
    return {"tick": 1, "self": {"hp": 20, "hunger": 20, "inventory": inv or {}}, "entities": list(entities),
            "events": list(events)}


def test_trust_moves_on_deeds_and_fades():
    n = Now("Rook", ("wood", "stone"))
    n.see(obs([{"type": "give", "from": "Mira", "to": "Rook", "items": {"wood": 2, "stone": 1}}]))
    assert round(n.trust("Mira"), 2) == N.TRUST["gave"]                    # (one gift, once, whatever it holds)
    assert n.balance("Mira") == -3 and "you owe them 3 items" in n.person("Mira")
    n.see(obs([{"type": "give", "from": "Rook", "to": "Mira", "items": {"wood": 5}}]))
    assert n.balance("Mira") == 2 and "they owe you 2 items" in n.person("Mira")
    n.see(obs([{"type": "attack", "attacker": "doben00", "target": "Rook", "damage": 1}]))
    assert n.trust("doben00") < 0 and "trust -0.4" in n.person("doben00")
    n.relations["doben00"]["trust_at"] = time.time() - N.TRUST_HALF_S       # (two days on)
    assert round(n.trust("doben00"), 2) == -0.2


def test_a_lacking_aim_names_who_had_it(tmp_path):
    m = Mind("Rook", "quiet", "http://x", FakeHTTP("{}"), tmp_path / "memory.json", Now("Rook", ("wood", "stone", "fiber")))
    m.aims = ["pickaxe"]
    seen = {"kind": "agent", "name": "doben00", "dx": 3, "dy": 0, "hp": 20, "holding": "stone"}
    m.now.see(obs(entities=[seen]))
    m.now.see(obs([{"type": "give", "from": "Mira", "to": "Rook", "items": {"fiber": 1}}]))
    facts = m.facts(obs(entities=[seen], inv={"wood": 2}))
    line = next(x for x in facts.splitlines() if x.startswith("What your aims lack"))
    assert "3 stone (doben00 (here))" in line and "1 fiber (Mira, trust +0.2, you owe them 1 item)" in line
    m.apply({"goal": "free"})                                              # (kept across sessions)
    assert "seen_with" in Mind("Rook", "", "", None, tmp_path / "memory.json").relations["doben00"]
