"""PPO in the sim, never blind: every --check steps, one episode on each held-out world, a [check] line with the
KPI (Crafter score) beside the random and do-nothing players; stop when the score has not beaten its best for
--patience checks, or at --hours. Trainer side:
    PYTHONPATH=src ~/minish-rl/.venv/bin/python -m megastructure.rl.train --run r1"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .evaluate import baselines, metrics, run
from .net import GridVec, HybridMaskablePPO, HybridPPO
from .vec import PipeVecEnv

ROOT = Path(__file__).resolve().parents[3] / "rl_runs"


class Check(BaseCallback):
    def __init__(self, a, out: Path, eval_env, base):
        super().__init__()
        self.a, self.out, self.eval_env, self.base = a, out, eval_env, base
        self.next, self.best, self.flat, self.t0 = a.check, -1.0, 0, time.time()
        self.train_eps = deque(maxlen=200)

    def log(self, line: str):
        print(line, flush=True)
        with open(self.out / "checks.log", "a") as f:
            f.write(line + "\n")

    def _on_step(self) -> bool:
        for d, info in zip(self.locals["dones"], self.locals["infos"]):
            if d and "achievements" in info:
                self.train_eps.append(info)
        if self.num_timesteps < self.next:
            return True
        self.next += self.a.check
        pol = self.model.policy
        ev = self.eval_env
        if self.a.mask:
            m = metrics(run(ev, lambda o: pol.predict(o, deterministic=False, action_masks=np.stack(ev.masks))[0]))
        else:
            m = metrics(run(ev, lambda o: pol.predict(o, deterministic=False)[0]))
        tr = metrics(list(self.train_eps)) if self.train_eps else {}
        sps = self.num_timesteps / (time.time() - self.t0)
        self.log(f"[check] step={self.num_timesteps} min={(time.time() - self.t0) / 60:.0f} sps={sps:.0f} "
                 f"score={m['score']} (random {self.base['random']['score']}, legal random {self.base.get('random_legal', {}).get('score')}, noop {self.base['noop']['score']}) "
                 f"ach={m['ach']} ticks={m['ticks']} deaths={m['deaths']}/{m['n']} crafts_distinct={m['crafts_distinct']} "
                 f"places={m['places']} fail%={m['fail_pct']} fights={m['fights']} hits={m['hits']} "
                 f"train_score={tr.get('score')} train_ach={tr.get('ach')}")
        self.log("        top: " + json.dumps(m["top"]))
        if m["score"] > self.best:
            self.best, self.flat = m["score"], 0
            self.model.save(self.out / "best.zip")
        else:
            self.flat += 1
        self.model.save(self.out / "last.zip")
        if self.flat >= self.a.patience:
            self.log(f"[stop] score flat for {self.flat} checks (best {self.best})")
            return False
        if time.time() - self.t0 > self.a.hours * 3600:
            self.log(f"[stop] time budget ({self.a.hours} h) used; best {self.best}")
            return False
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="r1")
    p.add_argument("--envs", type=int, default=6)
    p.add_argument("--eval-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=50_000_000)
    p.add_argument("--check", type=int, default=200_000)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--resume", default="")
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--mask", type=int, default=1, help="1: MaskablePPO with the legal-action mask")
    a = p.parse_args()
    torch.set_num_threads(2)
    out = ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    env, ev = PipeVecEnv(a.envs, rng_seed=hash(a.run) % 1000), PipeVecEnv(a.eval_envs)
    bfile = ROOT / "baselines.json"
    base = json.loads(bfile.read_text()) if bfile.exists() else {}
    if "random_legal" not in base:
        base = baselines(ev, ev.spec["n_actions"])
    bfile.write_text(json.dumps(base, indent=1))
    algo = HybridMaskablePPO if a.mask else HybridPPO
    if a.resume:
        model = algo.load(a.resume, env=env, device="cpu")
    else:
        model = algo("MultiInputPolicy", env, n_steps=512, batch_size=1024, n_epochs=4, gamma=a.gamma, gae_lambda=0.95,
                    learning_rate=3e-4, ent_coef=a.ent, clip_range=0.2, device="cpu", verbose=0,
                    policy_kwargs=dict(features_extractor_class=GridVec, net_arch=dict(pi=[256], vf=[256])))
    cb = Check(a, out, ev, base)
    cb.log(f"[start] run={a.run} envs={a.envs} actions={env.spec['n_actions']} random={base['random']['score']} noop={base['noop']['score']}")
    try:
        model.learn(total_timesteps=a.steps, callback=cb, reset_num_timesteps=not a.resume)
    finally:
        env.close()
        ev.close()


if __name__ == "__main__":
    main()
