"""Held-out worlds: one episode on each eval seed, for a policy, a random player and a do-nothing player. The KPI is
Crafter's score (the geometric mean of the achievement rates), with survival, distinct crafts and fights beside it.
Trainer side (python 3.12, ~/minish-rl/.venv)."""

import json
import math
import time

import numpy as np

from .vec import EVAL0, PipeVecEnv

TICK_HZ = 10                     # (the game's; this module runs in the trainer, without the game)

EVAL_SEEDS = tuple(range(EVAL0, EVAL0 + 48))
LIVE_LIKE_SEEDS = tuple(range(EVAL0 + 5000, EVAL0 + 5024))     # the second held-out set: starts laid out like the live save
NAMES: list = []                  # the achievements (rl/achieve.py, game side): the workers send the list (spec)


# r6's 66 achievements (the first set), to compare runs across the widening
R6_SET = tuple([f"collect_{x}" for x in ("wood", "fiber", "stone", "stone_block", "dirt", "berries", "seed", "fruit", "flower",
                                          "mushroom", "ore", "meat", "wheat", "egg", "milk", "hay")]
               + [f"craft_{x}" for x in ("workbench", "plank", "stone", "campfire", "torch", "rope", "axe", "pickaxe", "hoe",
                                         "spear", "cooked_meat", "bread", "ingot", "sword", "door", "fence", "chest",
                                         "wood_wall", "stone_wall", "bed", "ladder", "stairs", "path")]
               + [f"place_{x}" for x in ("workbench", "campfire", "chest", "bed", "door", "fence", "wood_wall", "stone_wall",
                                         "dirt", "stone_block", "plank", "path", "torch")]
               + [f"eat_{x}" for x in ("berries", "fruit", "meat", "cooked_meat", "mushroom", "bread", "egg", "milk")]
               + [f"defeat_{x}" for x in ("tusker", "pincer", "strawman", "chicken", "cow", "sporeling")] + ["survive_night"])


def names_from(venv: PipeVecEnv):
    NAMES[:] = venv.spec["achievements"]


def crafter_score(rates: dict, names=None) -> float:
    """Crafter's score: the geometric mean of the success rates (in %), each plus one."""
    xs = [rates.get(a, 0.0) * 100 for a in (names or NAMES)]
    return math.exp(sum(math.log(1 + x) for x in xs) / len(xs)) - 1


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
    rates = {a: sum(a in e["achievements"] for e in eps) / n for a in set(NAMES) | set(R6_SET)}
    attacked = sum(e["won"] + e["lost"] + e["fled"] for e in eps)
    return {"score": round(crafter_score(rates), 3), "score_r6set": round(crafter_score(rates, R6_SET), 3), "n": n,
            "ach": round(float(np.mean([len(e["achievements"]) for e in eps])), 2),
            "ticks": int(np.mean([e["ticks"] for e in eps])), "deaths": sum(e.get("deaths", int(e["died"])) for e in eps),
            "crafts_distinct": round(float(np.mean([e["distinct_crafts"] for e in eps])), 2),
            "crafts": round(float(np.mean([e["crafts"] for e in eps])), 1),
            "places": round(float(np.mean([e["places"] for e in eps])), 1),
            "places_distinct": round(float(np.mean([e.get("distinct_places", 0) for e in eps])), 2),
            "nights": round(float(np.mean([e.get("nights", 0) for e in eps])), 2),
            "chunks": round(float(np.mean([e.get("chunks", 0) for e in eps])), 1),
            "fail_pct": round(100 * sum(e["fails"] for e in eps) / max(1, sum(e["acts"] for e in eps)), 1),
            "visits": "/".join(f"{sum(e.get('visit_' + k, 0) for e in eps)}{k[0].upper()}" for k in ("won", "fled", "endured", "killed")),
            "goals": " ".join(f"{k}:{sum(e.get('goals', {}).get(k, [0, 0])[0] for e in eps)}/{sum(e.get('goals', {}).get(k, [0, 0])[1] for e in eps)}"
                              for k in ("gather", "craft", "give", "ask", "stay", "help", "avoid")),
            "social": round(float(np.mean([len(e.get("social", [])) for e in eps])), 2),
            "social_kinds": {k: round(sum(k in e.get("social", []) for e in eps) / n, 2) for k in ("greeted_someone", "answered_someone", "gave_item", "traded", "helped", "worked_near_player")},
            "fights": f"{sum(e['won'] for e in eps)}W/{sum(e['lost'] for e in eps)}L/{sum(e['fled'] for e in eps)}F of {attacked}",
            "hits": sum(e["hits"] for e in eps), "hit_neighbour": sum(e.get("hit_neighbour", 0) for e in eps),
            "hit_neighbour_unprovoked": sum(e.get("hit_neighbour_unprovoked", 0) for e in eps),
            "first_strikes": sum(e.get("first_strikes", 0) for e in eps),
            "fills": "/".join(str(round(sum(e.get(f"fill_{n}", 0) for e in eps) / max(1, len(eps)), 2)) for n in (4, 6, 8)),
            "stock": "/".join(str(round(sum(e.get(f"stock_{n}", 0) for e in eps) / max(1, len(eps)), 2)) for n in (4, 8)),
            "multi": round(sum(e.get("multi_place", 0) for e in eps) / max(1, len(eps)), 2),
            "near_min": round(sum(e.get("near_ticks", 0) for e in eps) / TICK_HZ / 60, 1),       # (all episodes)
            "fs_per_near_min": round(sum(e.get("first_strikes", 0) for e in eps)
                                     / max(1e-9, sum(e.get("near_ticks", 0) for e in eps) / TICK_HZ / 60), 3),
            "picks": f"back {sum(e.get('pick_back', 0) for e in eps)} / slot {sum(e.get('pick_slot', 0) for e in eps)}",
            "said": round(float(np.mean([e.get("said", 0) for e in eps])), 1),
            "top": {a: round(r, 2) for a, r in sorted(rates.items(), key=lambda kv: -kv[1]) if r > 0}}


def legal_random(venv, rng):
    return np.array([rng.choice(np.flatnonzero(m)) for m in venv.masks])


def baselines(venv: PipeVecEnv, n_actions: int, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    t = time.time()
    out = {"random": metrics(run(venv, lambda o: rng.integers(0, n_actions, venv.num_envs))),
           "random_legal": metrics(run(venv, lambda o: legal_random(venv, rng))),
           "noop": metrics(run(venv, lambda o: np.zeros(venv.num_envs, int)))}
    out["secs"] = round(time.time() - t)
    return out


if __name__ == "__main__":
    import sys
    v = PipeVecEnv(4)
    names_from(v)
    print(json.dumps(baselines(v, v.spec["n_actions"]), indent=1))
    v.close()
    sys.exit(0)
