"""The build judge (rl/judge.py) on the builds a save holds: every closed room a door or gate leads into, judged by
v1 (verdict), v2 (sense) and v3's rules (looks), as the reference the body's homes are read against (#31). Read from a COPY of a save, never the live
game's own file (it writes there): take one first, e.g.

    python3 -c "import sqlite3; s = sqlite3.connect('file:../nexel-worlds/saves/caves/world.sqlite?mode=ro', uri=True); \\
        d = sqlite3.connect('/tmp/copy/world.sqlite'); s.backup(d)"
    python tools/judge_builds.py /tmp/copy/world.sqlite --seed 12"""

import argparse
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from nexel_worlds.sim.core import Simulation
from nexel_worlds.world import CHUNK

from nexel_agent.rl import judge


def homes(sim, db: Path):
    """Each closed room a door or gate opens into, once: (x, y) a tile inside it beside its door. (By the bed, it
    found none: the live save had no bed, 2026-11-26)"""
    seen = set()
    for cx, cy in sqlite3.connect(db).execute("select cx, cy from chunks"):
        for i in range(CHUNK):
            for j in range(CHUNK):
                x, y = cx * CHUNK + i, cy * CHUNK + j
                if sim.world.block(x, y) not in ("door", "gate"):
                    continue
                for n in judge._beside((x, y)):
                    if judge.walkable(sim.world, *n) and judge.report(sim.world, *n)["closed"]:
                        key = frozenset(judge.inside(sim.world, *n))
                        if key not in seen:
                            seen.add(key)
                            yield n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("db", type=Path, help="a COPY of a save's world.sqlite")
    ap.add_argument("--seed", type=int, default=12, help="the world's seed (the live one: 12)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        sim = Simulation(a.db, seed=a.seed, event_log=Path(tmp) / "events.jsonl")
        rows = []
        for x, y in homes(sim, a.db):
            v1, v2, v3 = judge.verdict(sim.world, x, y), judge.sense(sim.world, x, y), judge.looks(sim.world, x, y)
            rows.append({"at": [x, y], "v1": f"{v1['grade']}/{v1['of']}", "v1_failed": v1["failed"],
                         "v2": f"{v2['grade']}/{v2['of']}", "v2_failed": v2["failed"], "facts": v2["facts"],
                         "v3": f"{v3['grade']}/{v3['of']}", "v3_failed": v3["failed"], "looks": v3["facts"]})
    for r in rows:
        print(json.dumps(r, default=list))
    closed = [r for r in rows if r["facts"].get("closed")]
    print(f"\n{len(rows)} closed rooms behind a door or gate")
    print("v1 grades:", dict(sorted(Counter(r["v1"] for r in rows).items())))
    print("v2 grades:", dict(sorted(Counter(r["v2"] for r in rows).items())))
    print("v2 failed, of the closed:", dict(Counter(k for r in closed for k in r["v2_failed"]).most_common()))
    print("v3 grades:", dict(sorted(Counter(r["v3"] for r in rows).items())))
    print("v3 failed:", dict(Counter(k for r in rows for k in r["v3_failed"]).most_common()))


if __name__ == "__main__":
    main()
