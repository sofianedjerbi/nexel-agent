"""One held-out episode with the exported policy, told as a timeline: when each achievement came, what it did
most, how it ended (game Python, numpy only):
    PYTHONPATH=src ~/megastructure/.venv/bin/python -m megastructure.rl.watch rl_runs/r2/policy.npz 1000003"""

import sys
from collections import Counter

import numpy as np

from .core import SimEnv
from .encode import action_name
from .npolicy import NumpyPolicy


def watch(policy: str, seed: int, mask=True) -> dict:
    pol, env, rng = NumpyPolicy(policy), SimEnv(), np.random.default_rng(0)
    o = env.reset(seed)
    did, timeline, done = Counter(), [], set()
    while True:
        i = pol.act(o, rng, o["mask"] if mask else None)
        did[action_name(i)] += 1
        o, r, term, trunc, info = env.step(i)
        for a in sorted(env.ledger.done - done):
            me = env.a
            timeline.append(f"{(env.sim.tick - env.t0) / 600:5.1f} min  {a}  (hp {me.hp:g}, at {me.x},{me.y})")
        done = set(env.ledger.done)
        if term or trunc:
            return {"timeline": timeline, "top_actions": did.most_common(12), "end": info,
                    "inventory": dict(env.a.inventory)}


if __name__ == "__main__":
    out = watch(sys.argv[1], int(sys.argv[2]))
    print("\n".join(out["timeline"]))
    print("top actions:", out["top_actions"])
    print("pack at the end:", out["inventory"])
    print("end:", out["end"])
