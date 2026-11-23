"""The trained policy in plain numpy, for the game's own Python (no torch there): the same network as rl/net.py
(GridVec, then SB3's policy MLP with tanh, then the action logits), from weights rl/export.py wrote (.npz)."""

import numpy as np


def conv2d(x, w, b, stride=1, pad=1):
    """x (C, H, W), w (O, C, k, k): a cross-correlation, as torch's Conv2d."""
    c, h, wd = x.shape
    o, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    ho, wo = (h + 2 * pad - k) // stride + 1, (wd + 2 * pad - k) // stride + 1
    cols = np.empty((c, k, k, ho, wo), x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, i, j] = xp[:, i:i + stride * ho:stride, j:j + stride * wo:stride]
    return (w.reshape(o, -1) @ cols.reshape(c * k * k, -1)).reshape(o, ho, wo) + b[:, None, None]


def relu(x):
    return np.maximum(x, 0)


class NumpyPolicy:
    def __init__(self, path):
        z = np.load(path)
        self.p = {k: z[k].astype(np.float32) for k in z.files}

    def logits(self, obs: dict) -> np.ndarray:
        p, f = self.p, "features_extractor."
        g = p[f + "emb.weight"][obs["grid"].astype(np.int64)].transpose(2, 0, 1)
        x = np.concatenate([g, obs["height"][None].astype(np.float32)], 0)
        x = relu(conv2d(x, p[f + "cnn.0.weight"], p[f + "cnn.0.bias"]))
        x = relu(conv2d(x, p[f + "cnn.2.weight"], p[f + "cnn.2.bias"], stride=2))
        x = relu(conv2d(x, p[f + "cnn.4.weight"], p[f + "cnn.4.bias"], stride=2))
        w = p[f + "vec.0.weight"]
        v = relu(w @ obs["vec"][:w.shape[1]] + p[f + "vec.0.bias"])      # (a policy from before the vector grew: its part)
        h = relu(p[f + "head.0.weight"] @ np.concatenate([x.reshape(-1), v]) + p[f + "head.0.bias"])
        i = 0
        while f"mlp_extractor.policy_net.{i}.weight" in p:
            h = np.tanh(p[f"mlp_extractor.policy_net.{i}.weight"] @ h + p[f"mlp_extractor.policy_net.{i}.bias"])
            i += 2
        return p["action_net.weight"] @ h + p["action_net.bias"]

    def act(self, obs: dict, rng: np.random.Generator | None = None, mask=None) -> int:
        """Sampled from the policy's distribution (as in training and eval); argmax with no rng. mask: the legal
        actions (encode.legal), as MaskablePPO saw them."""
        lg = self.logits(obs)
        if mask is not None and len(mask) > len(lg):                     # (actions added since it was trained: never)
            lg = np.concatenate([lg, np.full(len(mask) - len(lg), -1e8, lg.dtype)])
        if mask is not None:
            lg = np.where(mask, lg, -1e8)
        if rng is None:
            return int(lg.argmax())
        pr = np.exp(lg - lg.max())
        return int(rng.choice(len(pr), p=pr / pr.sum()))
