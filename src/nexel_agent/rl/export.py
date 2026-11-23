"""A checkpoint (SB3 .zip, trainer side) to the numpy weights the live body loads (.npz), checked against torch on
a few real observations from the sim:
    PYTHONPATH=src ~/minish-rl/.venv/bin/python -m megastructure.rl.export rl_runs/r1/best.zip rl_runs/r1/policy.npz"""

import sys

import numpy as np
import torch

from .net import HybridMaskablePPO, HybridPPO
from .npolicy import NumpyPolicy
from .vec import PipeVecEnv


def export(src: str, dst: str):
    try:
        model = HybridMaskablePPO.load(src, device="cpu")
    except Exception:
        model = HybridPPO.load(src, device="cpu")
    sd = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()
          if k.startswith(("features_extractor.", "mlp_extractor.policy_net", "action_net"))}
    np.savez(dst, **sd)
    v = PipeVecEnv(1, seeds=[12345])
    obs = v.reset()
    npol, worst = NumpyPolicy(dst), 0.0
    for _ in range(20):
        with torch.no_grad():
            t_obs = {k: torch.as_tensor(o) for k, o in obs.items()}
            dist = model.policy.get_distribution(t_obs)
            ref = dist.distribution.logits[0].numpy()
        mine = npol.logits({k: o[0] for k, o in obs.items()})
        worst = max(worst, float(np.abs((mine - mine.max()) - (ref - ref.max())).max()))
        obs, *_ = v.step(np.array([int(ref.argmax())]))
    v.close()
    assert worst < 1e-3, worst
    print(f"exported {len(sd)} tensors to {dst}; numpy vs torch logits within {worst:.2e}")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
