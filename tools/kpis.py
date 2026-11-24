"""The KPIs that track how the AI player changes, run after run: a few clear numbers, each the target of an issue,
kept in docs/kpis.jsonl (one row per check of a training run, and one per live scorecard) and shown as one table
in docs/kpis.md.

    python tools/kpis.py backfill                 # every [check] / [live-like] line of rl_runs/r*.out
    python tools/kpis.py live rl_runs/score/<run> # a live scorecard (tools/rl_live_scorecard.py)
    python tools/kpis.py show                     # docs/kpis.md from the rows

Body, in the sim (held-out worlds; from train.py's checks):
- score: Crafter's score, the geometric mean of the achievement rates - higher is better
- deaths: deaths over the 48 (lush) or 24 (live-like) held-out episodes - lower (#16)
- fs_near_min: first strikes on a neighbour per minute spent near one - lower (#13)
- goals_done: the share of the goals set that were done - higher (#4, #22)
- tools: spears, axes and pickaxes made per episode - higher (#5)
- repeat_crafts: the share of crafts that repeat one already made this life - lower (#6)
- rooms: the share of episodes with a closed room built - higher (#6)
Live (a QA copy, with the scripted visitors; tools/scorecard.py), beside the real people of the world:
- lines_min, reply_p90_s, fs_hour (first strikes on people an hour), repeat_crafts_pct, tools_made, fail_pct,
  asks_met (the visitor's asks it met)."""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS, PAGE = ROOT / "docs" / "kpis.jsonl", ROOT / "docs" / "kpis.md"
SIM = ("score", "deaths", "fs_near_min", "goals_done", "tools", "repeat_crafts", "rooms", "fail_pct")
LIVE = ("lines_min", "reply_p90_s", "fs_hour", "repeat_crafts_pct", "tools_made", "fail_pct", "asks_met")
LOWER = {"deaths", "fs_near_min", "repeat_crafts", "fail_pct", "fs_hour", "repeat_crafts_pct", "reply_p90_s"}


def num(pat, line, cast=float):
    m = re.search(pat, line)
    return cast(m.group(1)) if m else None


def sim_row(run: str, step, set_: str, line: str) -> dict:
    goals = [tuple(map(int, g)) for g in re.findall(r"\w+:(\d+)/(\d+)", num(r"goals=\[([^\]]*)\]", line, str) or "")]
    gear = num(r"gear=\[([^\]]*)\]", line, str) or ""
    tools = [float(x) for x in re.findall(r"(?:spear|axe|pickaxe) ([\d.]+)", gear)]
    return {"kind": "sim", "run": run, "step": step, "set": set_,
            "score": num(r"score=([\d.]+)", line), "deaths": num(r"deaths=(\d+)/", line, int),
            "fs_near_min": num(r"fs/near_min=([\d.]+)", line),
            "goals_done": round(sum(d for d, _ in goals) / sum(n for _, n in goals), 3) if goals and sum(n for _, n in goals) else None,
            "tools": round(sum(tools), 2) if tools else None,
            "repeat_crafts": num(r"repeat_crafts ([\d.]+)", gear), "rooms": num(r"rooms=([\d.]+)", line),
            "fail_pct": num(r"fail%=([\d.]+)", line)}


def backfill() -> list:
    out = []
    for f in sorted(ROOT.glob("rl_runs/r*.out"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        step = None
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("[check]"):
                step = num(r"step=(\d+)", line, int)
                out.append(sim_row(f.stem, step, "lush", line))
            elif line.startswith("[live-like]") and step is not None:
                out.append(sim_row(f.stem, step, "live-like", line))
    return out


def live_row(run_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT / "tools"))
    import scorecard as S
    who = next(p.name for p in run_dir.iterdir() if p.is_dir() and p.name != "saves")
    P = S.players(S.read(run_dir / "saves" / "qa" / "events.jsonl"))
    r, own = S.row(P[who]) if who in P else {}, S.own(run_dir / who)
    try:
        v = json.loads((run_dir / "visitor.json").read_text())
        asks = v.get("asks", [])
        met = f"{sum(a.get('got') is not None for a in asks)}/{len(asks)}"
    except (OSError, ValueError):
        met = None
    return {"kind": "live", "run": run_dir.name, "who": who, "model": "words" if (run_dir / who / "latency.jsonl").exists() else "silent",
            "minutes": r.get("active min"), "lines_min": r.get("lines / min"), "reply_p90_s": own.get("heard to sent s (p90)"),
            "fs_hour": r.get("first strikes on people / hour"), "repeat_crafts_pct": r.get("repeat crafts %"),
            "tools_made": r.get("tools or weapons made"), "fail_pct": own.get("failed actions %"), "asks_met": met}


def load() -> list:
    return [json.loads(x) for x in ROWS.read_text().splitlines()] if ROWS.exists() else []


def save(rows: list):
    ROWS.parent.mkdir(exist_ok=True)
    ROWS.write_text("".join(json.dumps(r) + "\n" for r in rows))


def show(rows: list) -> str:
    def cell(v):
        return "-" if v is None else f"{v:g}" if isinstance(v, float) else str(v)
    sim = [r for r in rows if r["kind"] == "sim"]
    intro, _, defs = __doc__.split("\n\n", 2)
    out = [intro, "", defs.rstrip('"'), "", f"_Made by `tools/kpis.py show`, {time.strftime('%Y-%m-%d')}. Arrows: which way is "
           "better._", ""]
    for s in ("lush", "live-like"):
        out += [f"## Body in the sim: {s} held-out worlds", "", "| run | step | " + " | ".join(
            f"{k} {'↓' if k in LOWER else '↑'}" for k in SIM) + " |", "|---|---|" + "---|" * len(SIM)]
        out += [f"| {r['run']} | {r['step']} | " + " | ".join(cell(r.get(k)) for k in SIM) + " |" for r in sim if r["set"] == s]
        out.append("")
    live = [r for r in rows if r["kind"] == "live"]
    out += ["## Live (QA copy, scripted visitors)", "", "| run | words | min | " + " | ".join(
        f"{k} {'↓' if k in LOWER else '↑'}" for k in LIVE) + " |", "|---|---|---|" + "---|" * len(LIVE)]
    out += [f"| {r['run']} | {r['model']} | {cell(r['minutes'])} | " + " | ".join(cell(r.get(k)) for k in LIVE) + " |" for r in live]
    return "\n".join(out) + "\n"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    rows = load()
    if cmd == "backfill":
        rows = [r for r in rows if r["kind"] != "sim"] + backfill()
    elif cmd == "live":
        row = live_row(Path(sys.argv[2]).resolve())
        rows = [r for r in rows if not (r["kind"] == "live" and r["run"] == row["run"])] + [row]
    if cmd in ("backfill", "live"):
        save(rows)
    PAGE.write_text(show(rows))
    print(PAGE.read_text())


if __name__ == "__main__":
    main()
