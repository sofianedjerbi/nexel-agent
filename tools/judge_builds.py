"""The build judge (rl/judge.py) on the builds a save holds: every closed room a door or gate leads into, judged by
v1 (verdict), v2 (sense) and v3's rules (looks), as the reference the body's homes are read against (#31). With the
save's events and a name, each room is the player's (a piece of its walls laid by them, and still standing) or the
real players' (#27: the live scorecard, tools/rl_live_scorecard.py, and its KPIs, tools/kpis.py).
It reads a COPY of a save, never the live game's own file (it writes there), and judges on a copy of that:

    python3 -c "import sqlite3; s = sqlite3.connect('file:../nexel-worlds/saves/caves/world.sqlite?mode=ro', uri=True); \\
        d = sqlite3.connect('/tmp/copy/world.sqlite'); s.backup(d)"
    python tools/judge_builds.py /tmp/copy/world.sqlite [--events <its events.jsonl> --who Rook]"""

import argparse
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from nexel_worlds.sim.core import Simulation
from nexel_worlds.world import CHUNK

from nexel_agent.rl import judge

PLAYERS = "players"           # the rooms nobody named laid: the world's real people's


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


def placed_by(events: Path, who: str) -> set:
    """The tiles where `who` laid a piece that closes a room (judge.ENCLOSE) and it still stands: the last change to
    each tile is theirs, laying it."""
    last = {}
    for line in events.read_text().splitlines():
        e = json.loads(line)
        if e.get("type") == "tile" and "x" in e:
            last[(e["x"], e["y"])] = e
    return {p for p, e in last.items() if e.get("by") == who and e.get("cause") == "place" and e.get("block") in judge.ENCLOSE}


def judged(db: Path, events: Path | None = None, who: str | None = None) -> list:
    """Every closed room of the save, judged: {"at", "by": who or PLAYERS, "v1", "v2", "v3" (grades), "of" (each's
    checks), the failed checks of each, the facts}. On a copy of `db` (opening a world may write to it)."""
    mine = placed_by(events, who) if events and who else set()
    seed = dict(sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute("select key, value from meta")).get("seed", 12)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "world.sqlite"
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as a, sqlite3.connect(copy) as b:
            a.backup(b)
        sim, rows = Simulation(copy, seed=int(seed), event_log=Path(tmp) / "events.jsonl"), []
        for x, y in homes(sim, copy):
            v1, v2, v3 = judge.verdict(sim.world, x, y), judge.sense(sim.world, x, y), judge.looks(sim.world, x, y)
            walls = {n for p in judge.inside(sim.world, x, y) for n in judge._beside(p)}
            rows.append({"at": [x, y], "by": who if walls & mine else PLAYERS,
                         "v1": v1["grade"], "v2": v2["grade"], "v3": v3["grade"], "of": [v1["of"], v2["of"], v3["of"]],
                         "v1_failed": v1["failed"], "v2_failed": v2["failed"], "v3_failed": v3["failed"],
                         "facts": v2["facts"], "looks": v3["facts"]})
    return rows


def builds(run_dir: Path, who: str) -> list:
    """A live scorecard's builds (rl_runs/score/<run>: its QA copy and events), judged once: kept in builds.json."""
    kept = run_dir / "builds.json"
    if kept.exists():
        return json.loads(kept.read_text())
    qa = run_dir / "saves" / "qa"
    rows = judged(qa / "world.sqlite", qa / "events.jsonl", who)
    kept.write_text(json.dumps(rows, default=list))
    return rows


def summary(rows: list, by: str) -> dict:
    """Of the rooms of `by`: how many, and the best grade by each judge (None: none)."""
    mine = [r for r in rows if r["by"] == by]
    best = lambda k: max((r[k] for r in mine), default=None)            # noqa: E731
    return {"builds": len(mine), "best_grade": best("v1"), "best_sense": best("v2"), "best_looks": best("v3")}


def table(rows: list, who: str) -> str:
    """The builds as a scorecard table: the player's rooms beside the real players'."""
    of = rows[0]["of"] if rows else [len(judge.CHECKS), len(judge.SENSE), len(judge.LOOKS)]
    cell = lambda v, n: "-" if v is None else f"{v}/{n}"                 # noqa: E731
    out = ["", "", "| builds (closed rooms, rl/judge.py) | rooms | best v1 (works) | best v2 (makes sense) | best v3 (looks) |",
           "|---|---|---|---|---|"]
    for by, name in ((who, who), (PLAYERS, "real players")):
        s = summary(rows, by)
        out.append(f"| {name} | {s['builds']} | {cell(s['best_grade'], of[0])} | {cell(s['best_sense'], of[1])} | "
                   f"{cell(s['best_looks'], of[2])} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("db", type=Path, help="a COPY of a save's world.sqlite")
    ap.add_argument("--events", type=Path, help="its events.jsonl: who laid what")
    ap.add_argument("--who", help="the player whose rooms are told apart from the real players'")
    a = ap.parse_args()
    rows = judged(a.db, a.events, a.who)
    for r in rows:
        print(json.dumps(r, default=list))
    print(f"\n{len(rows)} closed rooms behind a door or gate")
    for k in ("v1", "v2", "v3"):
        print(f"{k} grades:", dict(sorted(Counter(r[k] for r in rows).items())))
        print(f"{k} failed:", dict(Counter(c for r in rows for c in r[f"{k}_failed"]).most_common()))
    if a.who:
        print(table(rows, a.who))


if __name__ == "__main__":
    main()
