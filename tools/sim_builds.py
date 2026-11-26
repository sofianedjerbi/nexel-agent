"""A checkpoint's best builds in the sim (#33, lite): it plays held-out seeds (rl/evaluate.py's, the lush or the
live-like set) one episode each, in this process, and at each episode's end every closed room near the player and
its home is judged (tools/judge_builds.py: v1, v2, v3, and its tiles). The best of each episode goes to a JSON file
that tools/build_review.py --sim draws. It loads the Mac (torch, a sim): run it at nice 10, when no training runs.

    nice -n 10 python tools/sim_builds.py <checkpoint.zip> [--seeds 8] [--live-like] [--out sim_builds.json]"""

import argparse
import json
from pathlib import Path

from nexel_agent.rl import judge
from judge_builds import tiles

NEAR = 40                     # tiles round the player and its home a room of its is looked for in


def rooms(world, around) -> list:
    """Every closed room a door or gate opens into, within NEAR of the points `around`, judged: rows as
    judge_builds.judged gives them (by: the player)."""
    seen, out = set(), []
    for cx, cy in around:
        for x in range(cx - NEAR, cx + NEAR + 1):
            for y in range(cy - NEAR, cy + NEAR + 1):
                if world.block(x, y) not in ("door", "gate"):
                    continue
                for n in judge._beside((x, y)):
                    if not judge.walkable(world, *n) or not judge.report(world, *n)["closed"]:
                        continue
                    inside = judge.inside(world, *n)
                    if frozenset(inside) in seen:
                        continue
                    seen.add(frozenset(inside))
                    v1, v2, v3 = judge.verdict(world, *n), judge.sense(world, *n), judge.looks(world, *n)
                    out.append({"at": list(n), "by": "sim", "map": tiles(world, inside), "v1": v1["grade"], "v2": v2["grade"],
                                "v3": v3["grade"], "of": [v1["of"], v2["of"], v3["of"]], "v1_failed": v1["failed"],
                                "v2_failed": v2["failed"], "v3_failed": v3["failed"], "facts": v2["facts"], "looks": v3["facts"]})
    return out


def best(rows: list):
    return max(rows, key=lambda r: (r["v1"], r["v2"], r["v3"]), default=None)


def episode(pol, seed: int, env_args: dict) -> dict | None:
    """One held-out episode of the checkpoint; its best room (with its seed), or None."""
    import numpy as np
    from nexel_agent.rl.core import SimEnv
    dim, n_act = pol.observation_space["vec"].shape[0], pol.action_space.n     # (an older body: its own part, as ship_gate)
    env = SimEnv(**env_args)
    o, done, trunc = env.reset(seed), False, False
    while not (done or trunc):
        batch = {"grid": o["grid"][None], "height": o["height"][None], "vec": o["vec"][None, :dim]}
        a = pol.predict(batch, deterministic=False, action_masks=np.asarray(o["mask"])[None, :n_act])[0]
        o, _, done, trunc, _ = env.step(int(np.asarray(a).reshape(-1)[0]))
    around = [(env.a.x, env.a.y)] + ([tuple(env.social.home)] if env.social.home else [])
    b = best(rooms(env.sim.world, around))
    return {**b, "seed": seed} if b else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--seeds", type=int, default=8, help="how many held-out seeds (from the first)")
    ap.add_argument("--live-like", action="store_true", help="the live-like held-out set (thin_pct=100)")
    ap.add_argument("--out", type=Path, default=Path("sim_builds.json"))
    a = ap.parse_args()
    from nexel_agent.rl.evaluate import EVAL_SEEDS, LIVE_LIKE_SEEDS
    from nexel_agent.rl.net import HybridMaskablePPO, load
    pol = load(HybridMaskablePPO, a.checkpoint)
    seeds = (LIVE_LIKE_SEEDS if a.live_like else EVAL_SEEDS)[:a.seeds]
    rows = []
    for s in seeds:
        r = episode(pol, s, {"thin_pct": 100} if a.live_like else {})
        print(f"seed {s}: " + (f"v1 {r['v1']}, v2 {r['v2']}, v3 {r['v3']}" if r else "no closed room"), flush=True)
        rows += [r] if r else []
    a.out.write_text(json.dumps({"checkpoint": a.checkpoint, "seeds": list(seeds), "rooms": rows}, default=list))
    print(f"{len(rows)} of {len(seeds)} episodes closed a room: {a.out}")


if __name__ == "__main__":
    main()
