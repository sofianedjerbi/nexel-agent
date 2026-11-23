"""init_from carries the optimizer: after it, one update on a fixed batch moves the old weights as a plain resume
would (and a fresh Adam would not). Trainer side (torch): the game's suite skips it; run it with the trainer's Python:
    PYTHONPATH=src ~/minish-rl/.venv/bin/python tests/test_rl_adam.py"""

import copy
import tempfile
from pathlib import Path

try:
    import pytest
    torch, gym = pytest.importorskip("torch"), pytest.importorskip("gymnasium")
except ImportError:                     # (the trainer's Python has torch, not pytest: run as a script there)
    import gymnasium as gym
    import torch


class Box(gym.Env):
    def __init__(self, n_actions):
        self.observation_space = gym.spaces.Box(-1, 1, (6,))
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, a):
        return self.observation_space.sample(), 0.0, True, False, {}


OBS = torch.linspace(-1, 1, 6 * 8).reshape(8, 6)


def update(policy, n_old=3):
    """One optimizer step on a loss of the old actions' logits and the value only (the same function of the old weights)."""
    feats = policy.extract_features(OBS)
    pi, vf = policy.mlp_extractor(feats)
    loss = (policy.action_net(pi)[:, :n_old] * torch.arange(1, n_old + 1)).sum() + policy.value_net(vf).pow(2).sum()
    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()


def old_weights(policy, n_old=3):
    return {n: (p[:n_old] if n.startswith("action_net.") else p).detach().clone() for n, p in policy.named_parameters()}


def test_ported_adam_moves_like_a_resume():
    from stable_baselines3 import PPO
    from nexel_agent.rl.train import init_from
    torch.manual_seed(0)
    old = PPO("MlpPolicy", Box(3), device="cpu", seed=0)
    for _ in range(5):                                          # (an optimizer with a history)
        update(old.policy)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "old.zip")
        old.save(path)
        resumed = copy.deepcopy(old.policy)
        update(resumed)
        new = PPO("MlpPolicy", Box(4), device="cpu", seed=1)
        init_from(new, path, PPO, ["a3=a1"], ["a0", "a1", "a2", "a3"])
        fresh = copy.deepcopy(new.policy)
        fresh.optimizer = type(fresh.optimizer)(fresh.parameters(), **fresh.optimizer.defaults)
        update(new.policy)
        update(fresh)
    want, got, naive = old_weights(resumed), old_weights(new.policy), old_weights(fresh)
    worst = max((want[n] - got[n]).abs().max().item() for n in want)
    off = max((want[n] - naive[n]).abs().max().item() for n in want)
    assert worst < 1e-6, worst
    assert off > 10 * max(worst, 1e-9)                          # (without the port it would not match)


if __name__ == "__main__":
    test_ported_adam_moves_like_a_resume()
    print("ok")
