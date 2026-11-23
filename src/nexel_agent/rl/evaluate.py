"""Held-out worlds: one episode on each eval seed, for a policy, a random player and a do-nothing player. The KPI is
Crafter's score (the geometric mean of the achievement rates), with survival, distinct crafts and fights beside it.
Trainer side (python 3.12, ~/minish-rl/.venv)."""

import json
import time

import numpy as np

from .achieve import ACHIEVEMENTS, crafter_score
from .vec import EVAL0, PipeVecEnv

EVAL_SEEDS = tuple(range(EVAL0, EVAL0 + 16))


def run(venv: PipeVecEnv, act, seeds=EVAL_SEEDS) -> dict:
    """act(obs) -> actions. One episode per seed; returns the episodes' summaries."""
    venv.set_seeds(seeds)
    obs = venv.reset()
    want, eps = set(seeds), []
    while len(eps) < len(want):
        obs, _, done, infos = venv.step(act(obs))
        for d, info in zip(done, infos):
            if d and info.get("seed") in want:
                eps.append(info)
    return eps


def metrics(eps: list) -> dict:
    n = len(eps)
    rates = {a: sum(a in e["achievements"] for e in eps) / n for a in ACHIEVEMENTS}
    attacked = sum(e["won"] + e["lost"] + e["fled"] for e in eps)
    return {"score": round(crafter_score(rates), 3), "n": n,
            "ach": round(float(np.mean([len(e["achievements"]) for e in eps])), 2),
            "ticks": int(np.mean([e["ticks"] for e in eps])), "deaths": sum(e["died"] for e in eps),
            "crafts_distinct": round(float(np.mean([e["distinct_crafts"] for e in eps])), 2),
            "crafts": round(float(np.mean([e["crafts"] for e in eps])), 1),
            "places": round(float(np.mean([e["places"] for e in eps])), 1),
            "fail_pct": round(100 * sum(e["fails"] for e in eps) / max(1, sum(e["acts"] for e in eps)), 1),
            "fights": f"{sum(e['won'] for e in eps)}W/{sum(e['lost'] for e in eps)}L/{sum(e['fled'] for e in eps)}F of {attacked}",
            "hits": sum(e["hits"] for e in eps),
            "top": {a: round(r, 2) for a, r in sorted(rates.items(), key=lambda kv: -kv[1]) if r > 0}}


def baselines(venv: PipeVecEnv, n_actions: int, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    t = time.time()
    out = {"random": metrics(run(venv, lambda o: rng.integers(0, n_actions, venv.num_envs))),
           "noop": metrics(run(venv, lambda o: np.zeros(venv.num_envs, int)))}
    out["secs"] = round(time.time() - t)
    return out


if __name__ == "__main__":
    import sys
    v = PipeVecEnv(4)
    print(json.dumps(baselines(v, v.spec["n_actions"]), indent=1))
    v.close()
    sys.exit(0)
