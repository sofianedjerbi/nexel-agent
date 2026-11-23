"""What the forge learns from play: the players' own repeated failures (agents/*/decisions.jsonl) and the skills that
lost in play, turned into forge work: which curriculum skill to write again, with the failures quoted as facts.

    {"get_fiber": ["'dig_at bare_bush' failed 775x (not_found) while the goal was 'get fiber'", ...], ...}
"""

import json
from collections import Counter
from pathlib import Path

WORDS = {                                     # a failure's words -> the curriculum skill it is about
    "get_fiber": ("fiber", "bush", "thicket"), "get_wood": ("wood", "tree", "stump"), "get_stone": ("stone", "rock"),
    "make_workbench": ("workbench",), "make_axe": ("axe",), "make_pickaxe": ("pickaxe",), "make_hoe": ("hoe",),
    "make_campfire": ("campfire",), "make_wood_wall": ("wood_wall", "wall"), "make_door": ("door",),
    "make_chest": ("chest",),
}
MIN_FAILS = 20                                 # a command failing this often in a log is a lesson
TAIL = 20000                                   # the last records of each log


def tail(path: Path, n=TAIL) -> list[dict]:
    lines = path.read_text(errors="replace").splitlines()[-n:]
    out = []
    for x in lines:
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def skill_for(text: str) -> str | None:
    t = text.lower()
    for name in sorted(WORDS, key=lambda n: -max(len(w) for w in WORDS[n])):   # (pickaxe before axe)
        if any(w in t for w in WORDS[name]):
            return name
    return None


def lessons(data: Path, library_index: dict | None = None) -> dict[str, list[str]]:
    """skill -> facts from play that its next version should be written against."""
    out: dict[str, list[str]] = {}
    for log in sorted(Path(data).glob("*/decisions.jsonl")):
        fails = Counter((r.get("goal") or "", f"{r['tool']} {r['args'].get('text') or r['args'].get('item') or r['args'].get('recipe') or ''}".strip(),
                         r.get("error")) for r in tail(log) if "tool" in r and not r.get("ok") and r.get("source") == "model")
        for (goal, cmd, err), n in fails.most_common(10):
            if n < MIN_FAILS:
                break
            name = skill_for(f"{cmd} {goal}")
            if name:
                out.setdefault(name, []).append(f"'{cmd}' failed {n}x ({err}) while the goal was '{goal}' ({log.parent.name})")
    for name, m in (library_index or {}).items():
        if m.get("losses", 0) >= 3:
            out.setdefault(name, []).append(f"the skill lost {m['losses']} of {m['losses'] + m.get('wins', 0)} runs in play")
    return out
