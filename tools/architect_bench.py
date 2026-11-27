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

from nexel_agent.rl import architect, judge3d

BRIEFS = (("a home for one", 130), ("a small home for one by a river", 130), ("a cosy cabin for a farmer", 130),
          ("a home for two, a bed each", 140), ("a workshop home: a workbench by the door", 130),
          ("a round hut", 90), ("a round base for a camp: an open yard walled round, a campfire in it", 60),
          ("a two-storey house: a bedroom upstairs", 150), ("a watchtower, three storeys high", 170),
          ("a stone cottage", 150), ("a cottage under a gable roof of shingles with stone eaves", 140),
          ("a sod-roofed hut: earth on the roof", 150), ("a round house with a domed roof", 200),
          ("a home dug underground, hidden from the glooms", 82), ("a vegetable garden", 40),
          ("a house for one with its vegetable garden beside it", 180))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--rounds", type=int, default=architect.ROUNDS)
    ap.add_argument("--n", type=int, default=len(BRIEFS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--build", action="store_true", help="each design at full marks laid by the build verb in the sim")
    a = ap.parse_args()
    ask = architect.model(a.model_url)
    rows = []
    for brief, budget in BRIEFS[:a.n]:
        t0 = time.monotonic()
        d = architect.design(brief, ask, a.rounds, budget)
        s = d.get("structure")
        rows.append({"brief": brief, "s": round(time.monotonic() - t0), "budget": budget, **{k: d.get(k) for k in
                     ("grade", "of", "failed", "history", "rows", "spec", "error")}})
        if s is not None:
            rows[-1]["rows"] = [line for k in range(s.n) for line in (f"storey {k}:", *s.grid(k))]
        cost = s.cost() if s is not None else 0
        rows[-1]["cost"] = cost
        if a.build and d.get("spec") and d["grade"] == d["of"]:
            rows[-1]["built"] = judge3d.built(d["spec"])
        print(f"{d['grade']:>2}/{d['of']} {d['history']} {cost:>3}/{budget} built {rows[-1].get('built')}  {brief}",
              flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    full = [r for r in rows if r["grade"] == r["of"]]
    first = [r for r in rows if r["history"] and r["history"][0] == r["of"]]
    fails = Counter(c for r in rows for c in r["failed"] or [])
    print(f"\nfull marks {len(full)}/{len(rows)}, first drawing {len(first)}/{len(rows)}, "
          f"mean best {sum(r['grade'] for r in rows) / len(rows):.1f}/{rows[0]['of']}, "
          f"built whole {sum(1 for r in rows if (r.get('built') or {}).get('failed') == [])}, still failing: {dict(fails)}")
    for r in rows:
        print(f"\n{r['brief']} ({r['grade']}/{r['of']})\n" + "\n".join(r["rows"] or [r.get("error") or ""]))


if __name__ == "__main__":
    main()
