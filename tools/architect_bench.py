"""The architect's bench (#66): the model draws a home for each brief, the judge grades each round, and the table says
how often a first drawing passes, how many rounds the rest take, and which checks it keeps failing. The drawings go to
a JSON next to the table, for a look (they are text: a top-down grid each).

    python tools/architect_bench.py --model-url http://127.0.0.1:8080/v1/chat/completions --out ../local/architect/run1

Needs a local model server (one heavy job on the machine at a time: see CLAUDE.md)."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from nexel_agent.rl import architect

BRIEFS = ("a home for one", "a small home for one by a river", "a cosy cabin for a farmer",
          "a home for two, a bed each", "a workshop home: a workbench by the door",
          "a long narrow home", "the smallest home that still has everything", "a home with a big room to host friends",
          "a stone cottage", "a home that faces east (the door in the east wall)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--rounds", type=int, default=architect.ROUNDS)
    ap.add_argument("--n", type=int, default=len(BRIEFS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ask = architect.model(a.model_url)
    rows = []
    for brief in BRIEFS[:a.n]:
        t0 = time.monotonic()
        d = architect.design(brief, ask, a.rounds)
        rows.append({"brief": brief, "s": round(time.monotonic() - t0), **{k: d.get(k) for k in
                     ("grade", "of", "failed", "history", "rows", "error")}})
        print(f"{d['grade']:>2}/{d['of']} {d['history']} {brief}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    full = [r for r in rows if r["grade"] == r["of"]]
    first = [r for r in rows if r["history"] and r["history"][0] == r["of"]]
    fails = Counter(c for r in rows for c in r["failed"] or [])
    print(f"\nfull marks {len(full)}/{len(rows)}, first drawing {len(first)}/{len(rows)}, "
          f"mean best {sum(r['grade'] for r in rows) / len(rows):.1f}/{rows[0]['of']}, still failing: {dict(fails)}")
    for r in rows:
        print(f"\n{r['brief']} ({r['grade']}/{r['of']})\n" + "\n".join(r["rows"] or [r.get("error") or ""]))


if __name__ == "__main__":
    main()
