"""The ship gate (#7): a candidate body against the shipped one, on the same held-out worlds (48 lush, 24 laid out like
the live save), over several passes (the policy's own sampling seed changes, the worlds don't), each row a mean with
its 95% interval, and the difference with its interval. One verdict, by the rule of #16: a body ships when what the
run was for is better (--target) and score, deaths and first strikes are no worse.

    PYTHONPATH=src ~/minish-rl/.venv/bin/python tools/ship_gate.py rl_runs/r37/at_600k.zip rl_runs/r42/best.zip \
        --passes 4 --target goals_done

Trainer side (Python 3.12, torch). Writes the table to stdout and appends the verdict to rl_runs/LEDGER.md."""

import argparse
import math
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# (torch and the trainer's modules are imported where a body is played: what judges the numbers is plain Python, tested
# in the game's suite: tests/test_ship_gate.py)

T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 8: 2.36, 10: 2.26}     # Student's t, two-sided 95%, by n
LOWER_IS_BETTER = {"deaths", "fs_per_near_min", "repeat_crafts", "fail_pct"}
GUARDS = ("score", "deaths", "fs_per_near_min")


def rows(m: dict) -> dict:
    """The numbers a pass is judged on, from evaluate.metrics."""
    goals = dict(re.findall(r"(\w+):(\d+/\d+)", m["goals"]))
    done = [tuple(map(int, v.split("/"))) for v in goals.values()]
    top = m["top"]
    return {"score": m["score"], "deaths": m["deaths"], "fs_per_near_min": m["fs_per_near_min"],
            "crafts_distinct": m["crafts_distinct"], "repeat_crafts": m["repeat_crafts"], "fail_pct": m["fail_pct"],
            "tools_made": sum(top.get(f"craft_{t}", 0) for t in ("spear", "axe", "pickaxe", "sword", "hoe")),
            "rooms": top.get("room_built", 0), "build_grade": m.get("build_grade", 0), "builds_5": m.get("builds_5", 0),
            "goals_done": round(sum(d for d, _ in done) / max(1, sum(n for _, n in done)), 3),
            **{f"goal_{k}": round(int(v.split("/")[0]) / max(1, int(v.split("/")[1])), 3) for k, v in goals.items()}}


def trim(o: dict, masks, dim: int, n_act: int) -> tuple:
    """What a body from before the vector or the actions grew is shown: its own part of the vector and of the action
    masks, as npolicy plays it live. -> (the observation, the masks stacked)"""
    return {**o, "vec": o["vec"][:, :dim]}, np.stack(masks)[:, :n_act]


def passes(path: str, venvs: dict, n: int) -> dict:
    """{set: [rows of each pass]} for one checkpoint."""
    import torch
    from nexel_agent.rl.evaluate import metrics, run
    from nexel_agent.rl.net import HybridMaskablePPO, load
    pol, out = load(HybridMaskablePPO, path), {}
    dim, n_act = pol.observation_space["vec"].shape[0], pol.action_space.n

    def act(o, v):
        obs, masks = trim(o, v.masks, dim, n_act)
        return pol.predict(obs, deterministic=False, action_masks=masks)[0]
    for name, (venv, seeds) in venvs.items():
        out[name] = []
        for k in range(n):
            torch.manual_seed(k)
            np.random.seed(k)
            eps = run(venv, lambda o, v=venv: act(o, v), seeds)
            out[name].append(rows(metrics(eps)))
    return out


def ci(xs) -> tuple:
    m = statistics.fmean(xs)
    h = T95.get(len(xs), 2.0) * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("inf")
    return m, h


def verdict(k, base, cand) -> str:
    """better / same / worse: the interval of the difference (Welch, pass by pass) against zero."""
    (mb, hb), (mc, hc) = ci(base), ci(cand)
    d, h = mc - mb, math.hypot(hb, hc)
    if k in LOWER_IS_BETTER:
        d = -d
    return "better" if d - h > 0 else "worse" if d + h < 0 else "same"


def judge(B: dict, C: dict, targets) -> tuple:
    """The table's rows and the verdict, from each set's passes of the shipped body (B) and the candidate (C): it ships
    when every target row is better and no guard (GUARDS) is worse. -> ([table rows], ships)"""
    lines, ok = [], True
    for s in B:
        for k in B[s][0]:
            b, c = [r[k] for r in B[s]], [r[k] for r in C[s]]
            v = verdict(k, b, c)
            (mb, hb), (mc, hc) = ci(b), ci(c)
            lines.append(f"| {s} {k} | {mb:.3f} ± {hb:.3f} | {mc:.3f} ± {hc:.3f} | {v} |")
            ok &= not (k in GUARDS and v == "worse") and not (k in targets and v != "better")
    return lines, ok


def main():
    from nexel_agent.rl.evaluate import EVAL_SEEDS, LIVE_LIKE_SEEDS, names_from
    from nexel_agent.rl.vec import PipeVecEnv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--target", nargs="*", default=["score"], help="the rows the run was for: they must be better")
    ap.add_argument("--env", nargs="*", default=[], help="k=v for the eval envs (as in train.py)")
    a = ap.parse_args()
    t0 = time.time()
    ep = ["ep_ticks=9000", "death_ends=1", *a.env]           # (as train.py's held-out checks: 0.75 day, death ends it)
    lush = PipeVecEnv(a.envs, env_args=ep)
    like = PipeVecEnv(a.envs, env_args=[*ep, "thin_pct=100"])
    names_from(lush)
    venvs = {"lush": (lush, EVAL_SEEDS), "live-like": (like, LIVE_LIKE_SEEDS)}
    try:
        B, C = passes(a.base, venvs, a.passes), passes(a.cand, venvs, a.passes)
    finally:
        lush.close()
        like.close()
    rows_, ok = judge(B, C, a.target)
    lines = [f"| {a.passes} passes | {Path(a.base).parent.name}/{Path(a.base).name} | {Path(a.cand).parent.name}/{Path(a.cand).name} | |",
             "|---|---|---|---|", *rows_]
    out = "\n".join(lines) + f"\n\nverdict: {'SHIP' if ok else 'keep the shipped body'} (targets: {', '.join(a.target)}; " \
                             f"{round((time.time() - t0) / 60)} min)"
    print(out)
    with open(ROOT / "rl_runs" / "LEDGER.md", "a") as f:
        f.write(f"\n## gate {time.strftime('%Y-%m-%d %H:%M')}: {a.cand} vs {a.base}\n\n{out}\n")


if __name__ == "__main__":
    main()
