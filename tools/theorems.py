"""Checks of the research notes' own theorems in the game's simulator (nexel-website research/notes). Each prints
its rows and writes a CSV the notes plot.

    PYTHONPATH=src:tests nice -n 10 .venv/bin/python tools/theorems.py [bill|flight|all] [--out DIR]

bill    Theorem 1: built from exactly its bill (rl/project.py bill), with no gathering, does each design stand, and
        what is left over?
flight  Theorem 2: a flight laid with its door's column below standing, or with one wall of it missing: which storey
        does the game run it to?
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from nexel_agent.rl.build import laid_by, plan_of            # noqa: E402
from nexel_agent.rl.core import SimEnv                       # noqa: E402
from nexel_agent.rl.dug import design_of                     # noqa: E402
from nexel_agent.rl.plan import of_structure                 # noqa: E402
from nexel_agent.rl.project import BUILD, lacking            # noqa: E402

HOUSE = {"width": 3, "depth": 3, "door": "south", "roof": {"style": "gable"},
         "furniture": {"bed": [0, 0], "campfire": [2, 0], "chest": [0, 2], "chair": [2, 2]}}


def designs():
    from test_rl_structure import DESIGNS
    from test_rl_wings import WINGS
    return {"house": HOUSE, "two storeys": DESIGNS["two storeys"], "round home": DESIGNS["round home"],
            "tower": DESIGNS["tower"], "cross": {**WINGS["cross"], "roof": {"style": "hip", "material": "roof", "eaves": "plank"}}}


def env_for(seed):
    env = SimEnv(ep_ticks=10 ** 7, roam=0, rivals=0)
    env.reset(seed)
    env.folk = []
    env.goals = SimpleNamespace(goal=None, step=lambda *a: 0, draw=lambda *a, **k: None, counts={})
    env.a.inventory.clear()
    return env


def check_bill(seeds=(7, 9, 21)):
    rows = []
    for name, spec in designs().items():
        need = lacking(of_structure(design_of(spec)), lambda c: False, {})
        for seed in seeds:
            env = env_for(seed)
            env.social.plan = of_structure(design_of(spec))
            env.a.inventory.update(need)
            env.obs = env.sim.observe(env.a)
            for _ in range(4000):
                if not env.encoded()["mask"][BUILD]:
                    break
                env.step(BUILD)
            plan = plan_of(env.social)
            lb = laid_by(env.obs, env.social, plan) if env.social.home else (lambda c: False)
            left = sum(not lb(c) for c in plan.cells)
            spare = sum(env.a.inventory.get(k, 0) for k in need)
            rows.append([name, seed, need.get("wood", 0), need.get("stone", 0), need.get("fiber", 0), int(left == 0), left, spare])
            print("bill", rows[-1], flush=True)
    return ["design", "seed", "wood", "stone", "fiber", "whole", "cells_left", "raw_left"], rows


def check_flight():
    """Stamp the design whole but its stairs (judge3d.stamp), take out one wall of a flight's door column below its
    storey or none, and ask the game where a flight laid from the planned stand runs."""
    from nexel_agent.rl import judge3d
    rows = []
    for name in ("two storeys", "tower"):
        s = design_of(designs()[name])
        for top, steps, at, k in s.stairs:
            door = (2 * top[0] - steps[1][0], 2 * top[1] - steps[1][1])
            below = [c for c in s.cells() if c[:2] == door and 0 < c[2] < 3 * k and c[3] == "wall"]
            for missing in [None, *below]:
                env = env_for(5)
                a = env.a
                x0, y0 = a.x - 24, a.y - 24
                cells = [c for c in s.cells() if c != missing]
                lv = judge3d.stamp(env.sim, SimpleNamespace(cells=lambda cells=cells: cells, clear=s.clear), x0, y0)
                a.x, a.y = x0 + at[0][0], y0 + at[0][1]
                a.inventory["stairs"] = 20
                try:
                    run = env.sim.flight(a, x0 + top[0], y0 + top[1])
                    reached = (run[0][3] + 1 - lv) // 3
                except Exception as e:                   # (the game refuses: no flight)
                    reached = f"refused: {e}"
                rows.append([name, k, "none" if missing is None else f"wall z{missing[2]}", reached])
                print("flight", rows[-1], flush=True)
    return ["design", "storey", "missing", "storey_reached"], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("bill", check_bill), ("flight", check_flight)):
        if a.what in (name, "all"):
            header, rows = fn()
            with open(out / f"t_{name}.csv", "w", newline="") as f:
                w = csv.writer(f, delimiter=" ")
                w.writerow(header)
                w.writerows([[str(v).replace(" ", "_") for v in r] for r in rows])


if __name__ == "__main__":
    main()
