"""The trainer's side (torch, SB3, gymnasium: ~/minish-rl/.venv, Python 3.12): a VecEnv whose envs are workers in
the game's own Python (rl/worker.py), since the game needs 3.14. Train seeds are drawn at random below EVAL0;
held-out seeds start at EVAL0 and are never trained on."""

import os
import random
import subprocess
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .worker import read, write

GAME_PY = os.environ.get("GAME_PY", str(Path(__file__).resolve().parents[3] / ".venv/bin/python"))     # a Python with nexel-worlds installed
SRC = str(Path(__file__).resolve().parents[2])
EVAL0 = 1_000_000


class Worker:
    def __init__(self, args=()):
        env = dict(os.environ, PYTHONPATH=SRC, OMP_NUM_THREADS="1")
        self.p = subprocess.Popen([GAME_PY, "-m", "nexel_agent.rl.worker", *args], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, env=env)

    def send(self, *msg):
        write(self.p.stdin, msg)

    def recv(self):
        r = read(self.p.stdout)
        if r is None:
            raise RuntimeError("an env worker died")
        return r

    def close(self):
        try:
            self.send("close")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def spaces_of(spec):
    s = spec["side"]
    obs = spaces.Dict({"grid": spaces.Box(0, spec["n_chars"] - 1, (s, s), np.uint8),
                       "height": spaces.Box(-1, 1, (s, s), np.float32),
                       "vec": spaces.Box(-5, 5, (spec["vec"],), np.float32)})
    return obs, spaces.Discrete(spec["n_actions"])


class PipeVecEnv(VecEnv):
    def __init__(self, n: int, seeds=None, env_args=(), rng_seed=0):
        """seeds: a list to go through in order (eval); None: random train seeds."""
        self.workers = [Worker(env_args) for _ in range(n)]
        self.workers[0].send("spec")
        self.spec = self.workers[0].recv()
        self.rng = random.Random(rng_seed)
        self.set_seeds(seeds)
        self.actions = None
        super().__init__(n, *spaces_of(self.spec))

    def set_seeds(self, seeds):
        self.seeds = list(seeds) if seeds is not None else None

    def next_seed(self):
        if self.seeds is None:
            return self.rng.randrange(EVAL0)
        return self.seeds.pop(0) if self.seeds else EVAL0 - 1          # (spent: a world never scored)

    def _stack(self, obs_list):
        self.masks = [o.pop("mask") for o in obs_list]               # (legal actions: action_masks, not an input)
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

    def reset(self):
        self.playing = [self.next_seed() for _ in self.workers]       # the world each worker is in now
        for w, sd in zip(self.workers, self.playing):
            w.send("reset", sd)
        return self._stack([w.recv() for w in self.workers])

    def step_async(self, actions):
        for w, a in zip(self.workers, actions):
            w.send("step", int(a))

    def step_wait(self):
        out = [list(w.recv()) for w in self.workers]
        for i, o in enumerate(out):
            if o[2] or o[3]:                                   # the episode ended: on to the next world
                o[0].pop("mask", None)
                o[4]["terminal_observation"], o[4]["seed"] = o[0], self.playing[i]
                self.playing[i] = self.next_seed()
                self.workers[i].send("reset", self.playing[i])
                o[0] = self.workers[i].recv()
        obs = self._stack([o[0] for o in out])
        rew = np.array([o[1] for o in out], np.float32)
        done = np.array([o[2] or o[3] for o in out])
        infos = []
        for i, o in enumerate(out):
            info = dict(o[4])
            if o[3]:
                info["TimeLimit.truncated"] = True
            infos.append(info)
        return obs, rew, done, infos

    def close(self):
        for w in self.workers:
            w.close()

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            return list(self.masks)
        return [None] * self.num_envs

    def has_attr(self, attr_name):
        return attr_name == "action_masks"

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [None] * self.num_envs
