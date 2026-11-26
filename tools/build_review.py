"""The build review page (#33, lite: no renders): every closed room of the saved live runs drawn top-down as a grid of
its tiles, read from the world, beside the judge's three reports (rl/judge.py: v1 verdict, v2 sense, v3 looks;
tools/judge_builds.py). The player's rooms run by run; the real players' (the same in every QA copy) once, from the
last run. Drafts go outside the repos: the page is written to ~/nexel/local/builds/.

    python tools/build_review.py                       # every run in rl_runs/score
    python tools/build_review.py rl_runs/score/<run>   # some of them
    python tools/build_review.py --out <page.html>
    python tools/build_review.py --sim sim_builds.json # and a checkpoint's best sim builds (tools/sim_builds.py)"""

import argparse
import html
import json
import time
from pathlib import Path

from nexel_worlds.tiles import BLOCKS, GROUNDS

import judge_builds as J

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "nexel" / "local" / "builds" / "index.html"   # (~/nexel/local: drafts, beside the repos, in no worktree)
NAME = {b.char: b.name for b in BLOCKS} | {g.char: g.name for g in GROUNDS}
COLOUR = {   # (a tile's colour on the page: what stands on it, else its ground)
    "wood_wall": "#8b5a2b", "stone_wall": "#8a8f8c", "door": "#e0a84f", "gate": "#d9b06a", "fence": "#a77a4f",
    "glass": "#cfeaf2", "bed": "#d9675f", "chest": "#d4a93a", "campfire": "#ff8a1f", "fire": "#ff5a1f",
    "workbench": "#6e4325", "table": "#9b5b33", "chair": "#b8743f", "rug": "#b04a6a", "tree": "#2f6b34",
    "rock": "#6b6b6b", "ladder": "#c49a6c", "torch": "#ffd36b",
    "grass": "#8fc06a", "dirt": "#a67c52", "path": "#b5aa98", "cobble": "#b5aa98", "floor": "#c8a77a",
    "water": "#3b78c2", "mid_water": "#2e63a8", "deep_water": "#244f8a", "sand": "#e2cf8f", "ash": "#8c8c8c",
}


def who_of(run: Path) -> str:
    return next(p.name for p in run.iterdir() if p.is_dir() and p.name != "saves")


def grid(m: dict) -> str:
    """The room's tiles as a table: each cell the colour of what stands there (or its ground), roofed tiles hatched,
    the inside outlined; its name on hover."""
    out = ['<table class="g">']
    for j, row in enumerate(m["blocks"]):
        cells = []
        for i, b in enumerate(row):
            name = NAME.get(b, "?") if NAME.get(b, "none") != "none" else NAME.get(m["grounds"][j][i], "?")
            cls = ("r " if m["roofs"][j][i] == "1" else "") + ("in" if m["inside"][j][i] == "1" else "")
            cells.append(f'<td class="{cls}" style="background:{COLOUR.get(name, "#d0d0d0")}" title="{html.escape(name)} '
                         f'({m["x0"] + i}, {m["y0"] + j})"></td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(out) + "</table>"


def report(r: dict) -> str:
    of = r["of"]
    fails = lambda k: ", ".join(x.replace("_", " ") for x in r[k]) or "none"       # noqa: E731
    f = r["facts"]
    facts = [f"{f.get('area', '?')} tiles inside"] + ([f"core: {', '.join(f'{n} {k}' for k, n in f['core'].items())}"] if f.get("core") else [])
    return (f'<div class="rep"><p><b>v1 works {r["v1"]}/{of[0]}</b> · failed: {fails("v1_failed")}</p>'
            f'<p><b>v2 makes sense {r["v2"]}/{of[1]}</b> · failed: {fails("v2_failed")}</p>'
            f'<p><b>v3 looks {r["v3"]}/{of[2]}</b> · failed: {fails("v3_failed")}</p>'
            f'<p class="f">at {r["at"][0]}, {r["at"][1]} · {html.escape("; ".join(facts))}</p></div>')


def room(r: dict) -> str:
    return f'<div class="room">{grid(r["map"])}{report(r)}</div>'


def page(runs: list, sims: list = ()) -> str:
    """runs: [(run name, the player, its rooms judged)], oldest first; sims: tools/sim_builds.py's files, read."""
    parts = ['<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Build review</title><style>',
             "body{font:14px/1.45 system-ui,sans-serif;background:#1d1f24;color:#e8e6df;margin:0;padding:20px 28px 60px}",
             "h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:26px 0 8px;border-top:1px solid #333842;padding-top:12px}",
             ".note{color:#a9a6a0;max-width:900px}.room{display:flex;gap:18px;align-items:flex-start;margin:10px 0 16px}",
             "table.g{border-collapse:collapse}table.g td{width:16px;height:16px;padding:0;border:1px solid #1d1f24}",
             "td.in{outline:1px solid #fff4;outline-offset:-1px}",
             "td.r{background-image:repeating-linear-gradient(45deg,#0004 0 2px,transparent 2px 5px)!important}",
             ".rep p{margin:2px 0}.rep .f{color:#a9a6a0;font-size:13px}.key span{display:inline-block;width:12px;height:12px;"
             "vertical-align:middle;margin:0 4px 0 10px;border:1px solid #0006}</style></head><body>",
             "<h1>Build review: the AI player's homes, beside real players'</h1>",
             f'<p class="note">Every closed room of the saved live runs (rl_runs/score), drawn top-down tile by tile from the '
             f"world, beside the build judge's three reports (rl/judge.py). No renders: #33 lite. Made by tools/build_review.py, "
             f"{time.strftime('%Y-%m-%d %H:%M')}. Hatched: under a roof. Outlined: inside.</p>",
             '<p class="key">' + "".join(f'<span style="background:{c}"></span>{k.replace("_", " ")}' for k, c in
                                        list(COLOUR.items())[:12]) + "</p>"]
    for name, who, rows in runs:
        mine = [r for r in rows if r["by"] == who]
        parts.append(f"<h2>Run {html.escape(name)}: {html.escape(who)}'s rooms ({len(mine)})</h2>")
        parts += [room(r) for r in sorted(mine, key=lambda r: -(r["v1"] + r["v2"] + r["v3"]))] or \
            [f'<p class="note">{html.escape(who)} closed no room in this run.</p>']
    for sim in sims:
        rows = sorted(sim["rooms"], key=lambda r: -(r["v1"] + r["v2"] + r["v3"]))
        parts.append(f"<h2>In the sim: {html.escape(Path(sim['checkpoint']).name)}, the best room of each held-out episode "
                     f"({len(rows)} of {len(sim['seeds'])} closed one)</h2>")
        parts += [room(r).replace('<p class="f">at', f'<p class="f">seed {r["seed"]} · at', 1) for r in rows] or \
            ['<p class="note">None closed a room.</p>']
    if runs:
        name, who, rows = runs[-1]
        players = sorted((r for r in rows if r["by"] == J.PLAYERS), key=lambda r: -(r["v1"] + r["v2"] + r["v3"]))
        parts.append(f"<h2>Real players' rooms ({len(players)}), the reference (from run {html.escape(name)}'s copy)</h2>")
        parts += [room(r) for r in players] or ['<p class="note">None.</p>']
    return "\n".join(parts) + "\n</body></html>\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", type=Path, help="saved live runs (default: every one in rl_runs/score)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--sim", type=Path, action="append", default=[], help="a checkpoint's best sim builds (tools/sim_builds.py)")
    a = ap.parse_args()
    dirs = a.runs or sorted(p for p in (ROOT / "rl_runs" / "score").iterdir() if (p / "saves" / "qa" / "world.sqlite").exists())
    runs = []
    for d in dirs:
        who, qa = who_of(d), d / "saves" / "qa"
        runs.append((d.name, who, J.judged(qa / "world.sqlite", qa / "events.jsonl", who)))
        print(f"{d.name}: {sum(r['by'] == who for r in runs[-1][2])} of {who}'s, {len(runs[-1][2])} rooms in all", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page(runs, [json.loads(p.read_text()) for p in a.sim]))
    print(f"file://{a.out}")


if __name__ == "__main__":
    main()
