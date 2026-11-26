"""The talk bench (#10): ~40 fixed chat situations (tools/bench_talk.json) put to the live body's words path - rl/voice.py
Voice.line, with its prompt, its acts (rl/acts.py) and its deed check (rl/now.py backs) - against a model server, each
reply scored on rules a regular player's answer keeps:

- act: the words model labels the line heard as the case says (an ask for the item named, a greeting...)
- deed: no deed it never did (the claim the deed check drops, or what the case says it may not say)
- dump: no reading out the pack, unless it was asked what it carries
- style: lowercase, at most 15 words, no exclamation spam
- promise: an ask it can meet is met with a promise of that item; no promise otherwise
- reply: an answer where one is owed (an ask, a how); silence is fine elsewhere
- answer: a how names something the game's recipe takes

    PYTHONPATH=src python tools/bench_talk.py --url http://127.0.0.1:8080/v1/chat/completions
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from nexel_worlds.tiles import RECIPES

from nexel_agent.rl.acts import pack_dump
from nexel_agent.rl.encode import PACK
from nexel_agent.rl.now import Now
from nexel_agent.rl.voice import Voice

CASES = Path(__file__).with_name("bench_talk.json")
NAME, PERSONA = "Rook", "a quiet builder who likes to keep busy and help out when asked"
RULES = ("act", "deed", "dump", "style", "promise", "reply", "answer")
MAX_WORDS = 15


def load(path=CASES) -> list:
    """The cases, each with the defaults filled in."""
    d = json.loads(Path(path).read_text())
    return [{**d["defaults"], **c} for c in d["cases"]]


def observation(case) -> dict:
    """What the body sees in the case: itself, its pack, the people near it, and the lines just said."""
    near = [{"id": 10 + i, "kind": "agent", "name": n, "dx": 1 + i, "dy": 0, "hp": 20, "holding": "nothing", "doing": "idle"}
            for i, n in enumerate(case["near"])]
    return {"tick": 100, "time": "day 2, 10:00",
            "self": {"name": NAME, "x": 0, "y": 0, "hp": 18, "hunger": 14, "inventory": dict(case["pack"])},
            "entities": near, "events": [{"type": "say", "from": w, "text": t} for w, t in case["heard"]]}


def wanted(case) -> tuple:
    """The act asked for: (kinds allowed, item or None)."""
    act = case.get("act")
    if isinstance(act, list):
        return tuple(act), None
    kind, _, item = str(act).partition(" ")
    return (kind,), item or None


def score(case, said: str, labels: list, promise, false_deed: bool) -> dict:
    """Each rule -> True (kept), False (broken) or None (not asked of this case)."""
    kinds, item = wanted(case)
    label = labels[-1] if labels else (None, None)
    names = {NAME.lower()} | {w.lower() for w, _ in case["heard"]} | {n.lower() for n in case["near"]}
    words = said.split()
    caps = sum(ch.isupper() for w in words if w.strip(".,!?'").lower() not in names for ch in w)
    # (an ask's or an offer's item is what the body acts on (rl/live.py on_heard); a how's is not read: "how" alone is
    # right, a how naming another item is not. The 30B wrote "how" alone for 4 of the 6)
    named = label[1] == item or (label[0] == "how" and label[1] is None)
    out = {"act": label[0] in kinds and (item is None or named),
           "deed": not false_deed and not (said and case.get("must_not") and re.search(case["must_not"], said.lower())),
           "dump": kinds == ("pack",) or not pack_dump(said, PACK),
           "style": None if not said else len(words) <= MAX_WORDS and caps <= 1 and "!!" not in said,
           "reply": bool(said) if case["reply"] == "required" else None,
           "answer": None}
    if case.get("promise"):
        out["promise"] = bool(promise) and promise[0] == case["promise"]
    else:
        out["promise"] = promise is None
    if kinds == ("how",) and item in RECIPES:
        takes = {k.replace("_", " ") for k in RECIPES[item].inputs} | set(RECIPES[item].inputs)
        out["answer"] = any(t in said.lower() for t in takes)
    return out


async def run_case(case, url, http) -> dict:
    """The case through Voice.line, as the live body answers: -> the line, its labels and promise, and the score."""
    now = Now(NAME, PACK)
    for who, rec in (case.get("record") or {}).items():
        now.relations[who] = {"record": dict(rec)}
    now.see(observation(case))
    v = Voice(NAME, PERSONA, url, http, os.devnull, now)
    logs, labels = [], []
    v.write, v.on_heard = logs.append, lambda who, kind, item: labels.append((kind, item))
    said = await v.line([(w, t, time.monotonic()) for w, t in case["heard"]], case["doing"])
    rec = next(r for r in logs if "acts" in r)
    dropped = next((r for r in logs if r.get("kind") == "dropped"), None)
    said = said or (dropped or {}).get("said", "")                # (a line the deed check held back is still the model's)
    return {"id": case["id"], "said": said, "acts": rec.get("acts"), "promise": rec.get("promise"), "error": rec.get("error"),
            "llm_s": rec.get("llm_s"), **score(case, said, labels, rec.get("promise"), dropped is not None)}


async def bench(url, http, cases) -> list:
    return [await run_case(c, url, http) for c in cases]


def table(rows) -> str:
    """One line a case (. kept, X broken, blank not asked), then each rule's share kept and the issue's targets."""
    mark = {True: ".", False: "X", None: " "}
    out = [f"{'case':18} " + " ".join(f"{r[:6]:6}" for r in RULES) + "  said"]
    for r in rows:
        out.append(f"{r['id'][:18]:18} " + " ".join(f"{mark[r[k]]:6}" for k in RULES) + f"  {r['error'] or r['said']!r}"[:90])
    kept = {k: [r[k] for r in rows if r[k] is not None] for k in RULES}
    out.append(f"{'kept':18} " + " ".join(f"{(f'{sum(v)}/{len(v)}' if v else '-'):6}" for v in kept.values()))
    dumps = sum(r["dump"] is False for r in rows)
    deeds = sum(kept["deed"]) / max(1, len(kept["deed"]))
    out.append(f"inventory dumps {dumps} (target 0), true to deeds {deeds:.0%} (target 90%), "
               f"all rules kept {sum(all(r[k] is not False for k in RULES) for r in rows)}/{len(rows)}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--json", help="also write each case's result here (JSON lines)")
    args = ap.parse_args(argv)
    import httpx

    async def go():
        async with httpx.AsyncClient() as http:
            return await bench(args.url, http, load(args.cases))
    rows = asyncio.run(go())
    print(table(rows))
    if args.json:
        Path(args.json).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return 0 if all(r["error"] is None for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
