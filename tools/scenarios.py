"""Scenario benchmark: the body on tasks other than building, in the game's simulator, from an empty pack, the
goal set as the language model sets it live. Two controllers on the same seeds (paired):

    learned   the shipped policy (models/rook.npz) chooses every action, the goal in its input, as live
    skills    the goal's verb (pursue: gather, craft) whenever it has a step; the policy otherwise

A run succeeds when the goal is done within the decision budget, read from the pack (gather: n more of the item;
craft: one made). Building runs through the project executor (rl/project.py). Reports the success share with a
Wilson 95% interval and the decisions to success (median), per scenario and controller.

    PYTHONPATH=src nice -n 10 .venv/bin/python tools/scenarios.py [--seeds 10] [--budget 600] [--out scenarios.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexel_agent.rl.core import SimEnv                       # noqa: E402
from nexel_agent.rl.goals import Goal                        # noqa: E402
from nexel_agent.rl.npolicy import NumpyPolicy                # noqa: E402
from nexel_agent.rl.project import BUILD, PURSUE, Project    # noqa: E402

POLICY = Path(__file__).resolve().parents[1] / "models" / "rook.npz"
SCENARIOS = {                                  # name: (goal kind, item, how many more in the pack)
    "gather wood": ("gather", "wood", 10), "gather stone": ("gather", "stone", 10),
    "gather fiber": ("gather", "fiber", 5), "gather berries": ("gather", "berries", 5),
    "craft workbench": ("craft", "workbench", 1), "craft axe": ("craft", "axe", 1),
    "craft bed": ("craft", "bed", 1), "craft door": ("craft", "door", 1),
}
HOUSE = {"width": 3, "depth": 3, "door": "south", "roof": {"style": "gable"},
         "furniture": {"bed": [0, 0], "campfire": [2, 0], "chest": [0, 2], "chair": [2, 2]}}


def env_for(seed):
    env = SimEnv(ep_ticks=10 ** 7, roam=0, rivals=0)
    env.reset(seed)
    env.folk = []
    env.goals = SimpleNamespace(goal=None, step=lambda *a: 0, draw=lambda *a, **k: None, counts={})
    env.a.inventory.clear()
    env.obs = env.sim.observe(env.a)
    return env


def run_goal(pol, kind, item, n, seed, controller, budget):
    env, rng = env_for(seed), np.random.default_rng(seed)
    env.goals.goal = Goal(kind, item, None, env.obs["tick"], 0)
    for d in range(1, budget + 1):
        enc = env.encoded()
        mask = enc["mask"]
        i = PURSUE if controller == "skills" and mask[PURSUE] else pol.act(enc, rng, mask)
        env.step(i)
        if env.a.inventory.get(item, 0) >= n:
            return d
        if env.a.hp <= 0 or getattr(env.a, "deaths", 0):
            return None
    return None


def run_build(seed, budget):
    from nexel_agent.rl.build import laid_by, plan_of
    from nexel_agent.rl.dug import design_of
    from nexel_agent.rl.plan import of_structure
    env = env_for(seed)
    env.social.plan = of_structure(design_of(HOUSE))
    p = Project()
    for d in range(1, budget + 1):
        a = p.act(env.obs, env.social, bool(env.encoded()["mask"][BUILD]))
        if a is None:
            break
        env.goals.goal = a[1]
        env.step(a[0])
    if env.social.home is None:
        return None
    plan = plan_of(env.social)
    return d if all(laid_by(env.obs, env.social, plan)(c) for c in plan.cells) else None


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def summary(ds):
    ok = [d for d in ds if d is not None]
    return {"success": len(ok), "of": len(ds), "wilson95": wilson(len(ok), len(ds)),
            "median_decisions": float(np.median(ok)) if ok else None, "runs": ds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--out", default="scenarios.json")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    pol = NumpyPolicy(str(POLICY))
    seeds = [1000 + k for k in range(a.seeds)]
    out = {}
    for name, (kind, item, n) in SCENARIOS.items():
        if a.only and a.only not in name:
            continue
        out[name] = {c: summary([run_goal(pol, kind, item, n, s, c, a.budget) for s in seeds]) for c in ("learned", "skills")}
        print(name, {c: (v["success"], v["of"], v["median_decisions"]) for c, v in out[name].items()}, flush=True)
    if not a.only or "build" in a.only:
        out["build house"] = {"project": summary([run_build(s, 4000) for s in seeds])}
        print("build house", out["build house"]["project"]["success"], flush=True)
    Path(a.out).write_text(json.dumps({"seeds": seeds, "budget": a.budget, "results": out}, indent=1))


if __name__ == "__main__":
    main()
