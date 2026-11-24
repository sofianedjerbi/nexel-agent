"""The policy's eyes (trainer side, torch): the tile grid through an embedding and a small CNN, with the height
map as one more channel; the body/pack vector through an MLP. One network for every player: what one learns, all have."""

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GridVec(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, emb: int = 12, out: int = 384):
        super().__init__(observation_space, features_dim=out)
        n_chars = int(observation_space["grid"].high.max()) + 1
        side = observation_space["grid"].shape[0]
        self.emb = nn.Embedding(n_chars, emb)
        self.cnn = nn.Sequential(
            nn.Conv2d(emb + 1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, emb + 1, side, side)).shape[1]
        n_vec = observation_space["vec"].shape[0]
        self.vec = nn.Sequential(nn.Linear(n_vec, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(flat + 128, out), nn.ReLU())

    def forward(self, obs):
        g = self.emb(obs["grid"].long()).permute(0, 3, 1, 2)
        x = torch.cat([g, obs["height"].unsqueeze(1)], 1)
        return self.head(torch.cat([self.cnn(x), self.vec(obs["vec"])], 1))


POLICY_KW = dict(features_extractor_class=GridVec, net_arch=dict(pi=[256], vf=[256]))


def load(algo, path, custom_objects=None, **kw):
    """A checkpoint, whatever package it was saved from: its policy kwargs are given here, not unpickled (a run from
    before the move to nexel_agent names megastructure.rl.net's extractor (the game's first package name), which is gone)."""
    return algo.load(path, device="cpu", custom_objects={"policy_kwargs": POLICY_KW, **(custom_objects or {})}, **kw)


def _move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


class OnMPS:
    """Acts on the CPU (one small batch a step), trains on MPS (batched backprop): as in ~/minish-rl."""

    train_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def train(self):
        cpu = self.device
        self.policy.to(self.train_device)
        _move_optimizer(self.policy.optimizer, self.train_device)
        self.device = self.rollout_buffer.device = self.train_device
        try:
            super().train()
        finally:
            self.policy.to(cpu)
            _move_optimizer(self.policy.optimizer, cpu)
            self.device = self.rollout_buffer.device = cpu


class HybridPPO(OnMPS, PPO):
    pass


class HybridMaskablePPO(OnMPS, MaskablePPO):
    """With the legal-action mask (encode.legal): the game's refusals are never sampled."""
