"""PPO in the sim, never blind: every --check steps, one episode on each held-out world, a [check] line with the
KPI (Crafter score) beside the random and do-nothing players; stop when the score has not beaten its best for
--patience checks, or at --hours. Trainer side:
    PYTHONPATH=src ~/minish-rl/.venv/bin/python -m nexel_agent.rl.train --run r1"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .evaluate import baselines, metrics, names_from, run
from .net import POLICY_KW, HybridMaskablePPO, HybridPPO, load
from .vec import PipeVecEnv

ROOT = Path(__file__).resolve().parents[3] / "rl_runs"


def gear(m: dict) -> str:
    """Tools in a check line: spears, axes and pickaxes made per episode (first-time rates), the share of blows struck
    armed, armour worn, and the share of moments by a bench with fiber still in the pack."""
    t = m["top"]
    return (f"gear=[spear {t.get('craft_spear', 0)} axe {t.get('craft_axe', 0)} pickaxe {t.get('craft_pickaxe', 0)} "
            f"armed_blows {m['armed']} armor {t.get('wear_armor', 0)} fiber_at_bench {m['bench_fiber']}]")


class Check(BaseCallback):
    def __init__(self, a, out: Path, eval_env, base):
        super().__init__()
        self.a, self.out, self.eval_env, self.base = a, out, eval_env, base
        self.next, self.best, self.flat, self.t0 = a.check, -1.0, 0, time.time()
        self.train_eps = deque(maxlen=200)
        self.tool_p = []                  # in training, where a tool craft is possible: the chance the policy gives it
        self.kls, self.ents = [], []      # approx KL, and the policy's entropy, of each update since the last check (the drift per update)

    def _on_rollout_start(self):
        v = self.model.logger.name_to_value
        if "train/entropy_loss" in v:
            self.ents.append(-float(v["train/entropy_loss"]))
        if "train/approx_kl" in v:
            self.kls.append(float(v["train/approx_kl"]))
            with open(self.out / "kl.log", "a") as f:
                f.write(json.dumps({"step": self.num_timesteps, "approx_kl": round(self.kls[-1], 5),
                                    "clip_fraction": round(float(v.get("train/clip_fraction", 0)), 4)}) + "\n")

    def log(self, line: str):
        print(line, flush=True)
        with open(self.out / "checks.log", "a") as f:
            f.write(line + "\n")

    def _on_training_start(self):
        names = self.training_env.spec["names"]
        self.tools = [names.index(n) for n in ("craft_spear", "craft_axe", "craft_pickaxe")]
        self.s0, self.next = self.num_timesteps, self.num_timesteps + self.a.check     # (resumed: counted from here)
        if self.a.eval_first:
            self.next = self.num_timesteps

    def _on_step(self) -> bool:
        for d, info in zip(self.locals["dones"], self.locals["infos"]):
            if d and "achievements" in info:
                self.train_eps.append(info)
        masks = self.locals.get("action_masks")
        if masks is not None and (rows := masks[:, self.tools].any(1)).any():
            with torch.no_grad():
                probs = self.model.policy.get_distribution(self.locals["obs_tensor"], action_masks=masks).distribution.probs
            self.tool_p += probs[torch.as_tensor(rows)][:, self.tools].sum(1).tolist()
        if self.a.ent_end is not None:
            done = min(1.0, (self.num_timesteps - self.s0) / max(1, self.a.steps))
            self.model.ent_coef = self.a.ent + (self.a.ent_end - self.a.ent) * done
        if self.num_timesteps < self.next:
            return True
        self.next = self.num_timesteps + self.a.check
        pol = self.model.policy
        ev = self.eval_env
        if self.a.mask:
            m = metrics(run(ev, lambda o: pol.predict(o, deterministic=False, action_masks=np.stack(ev.masks))[0]))
        else:
            m = metrics(run(ev, lambda o: pol.predict(o, deterministic=False)[0]))
        tr = metrics(list(self.train_eps)) if self.train_eps else {}
        sps = (self.num_timesteps - self.s0) / (time.time() - self.t0)
        self.log(f"[check] step={self.num_timesteps} min={(time.time() - self.t0) / 60:.0f} sps={sps:.0f} "
                 f"score={m['score']} r6set={m['score_r6set']} (random {self.base['random']['score']}, legal random {self.base.get('random_legal', {}).get('score')}, noop {self.base['noop']['score']}) "
                 f"ach={m['ach']} ticks={m['ticks']} deaths={m['deaths']}/{m['n']} crafts_distinct={m['crafts_distinct']} "
                 f"places={m['places']} places_distinct={m['places_distinct']} nights={m['nights']} chunks={m['chunks']} fail%={m['fail_pct']} fights={m['fights']} rival={m['visits']} social={m['social']}/6 hit_neighbour={m['hit_neighbour']} unprovoked={m['hit_neighbour_unprovoked']} first_strikes={m['first_strikes']} near_min={m['near_min']} fs/near_min={m['fs_per_near_min']} attack_picks=[{m['picks']}] said/ep={m['said']} goals=[{m['goals']}] hits={m['hits']} "
                 f"kl_mean={np.mean(self.kls) if self.kls else 0:.4f} kl_max={max(self.kls, default=0):.4f} entropy={np.mean(self.ents) if self.ents else 0:.3f} "
                 f"tool_p={np.mean(self.tool_p) if self.tool_p else 0:.4f} (n={len(self.tool_p)}) "
                 f"{gear(m)} fills={m.get('fills', '')} stock={m.get('stock', '')} multi={m.get('multi', '')} rooms={m['top'].get('room_built', 0)} "
                 f"train_score={tr.get('score')} train_ach={tr.get('ach')}")
        self.log("        top: " + json.dumps(m["top"]))
        self.log("        social: " + json.dumps(m["social_kinds"]))
        if getattr(ev, "live_like", None) is not None:
            from .evaluate import LIVE_LIKE_SEEDS
            lk = metrics(run(ev.live_like, lambda o: pol.predict(o, deterministic=False, action_masks=np.stack(ev.live_like.masks))[0],
                            LIVE_LIKE_SEEDS))
            self.log(f"[live-like] score={lk['score']} ach={lk['ach']} deaths={lk['deaths']}/{lk['n']} crafts_distinct={lk['crafts_distinct']} "
                     f"first_strikes={lk['first_strikes']} multi={lk['multi']} fills={lk['fills']} stock={lk['stock']} rooms={lk['top'].get('room_built', 0)} near_min={lk['near_min']} fs/near_min={lk['fs_per_near_min']} fail%={lk['fail_pct']} {gear(lk)}")
        self.kls, self.ents, self.tool_p = [], [], []
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


def init_from(model, path, algo, copy=(), names=()):
    """An older checkpoint's weights (as ~/minish-rl does): inputs that grew keep the old columns, the new ones start
    at zero; new actions get zero weights and a bias under every old one (rare at first, but tried)."""
    old_model = load(algo, path)
    old = old_model.policy.state_dict()
    new = model.policy.state_dict()
    for k, v in old.items():
        if k not in new:
            continue
        if new[k].shape == v.shape:
            new[k] = v
        elif v.dim() == 2 and new[k].shape[0] == v.shape[0] and new[k].shape[1] > v.shape[1]:
            new[k] = torch.zeros_like(new[k])
            new[k][:, :v.shape[1]] = v
        elif k.startswith("action_net.") and new[k].shape[0] > v.shape[0] and new[k].shape[1:] == v.shape[1:]:
            grown = torch.zeros_like(new[k])
            grown[:v.shape[0]] = v
            if v.dim() == 1:
                grown[v.shape[0]:] = v.min() - 1.0
            new[k] = grown
        print(f"init_from: {k} {tuple(v.shape)} -> {tuple(new[k].shape)}")
    for pair in copy:                                     # a new action that starts as an old one: same row of the head
        to, frm = (names.index(n) for n in pair.split("="))
        for k in ("action_net.weight", "action_net.bias"):
            new[k][to] = new[k][frm]
        print(f"init_from: {pair} (row {frm} -> {to})")
    model.policy.load_state_dict(new)
    adam_from(model, old_model, copy, names)


def adam_from(model, old_model, copy=(), names=()):
    """The old optimizer's state too (Adam's moments and step): a fresh Adam moves every weight by about lr at first,
    whatever its history (r29/r30 against r30c). Grown weights keep the old state in their old part, zero elsewhere,
    and a copied action row takes its source row's state."""
    o_idx = {n: i for i, (n, _) in enumerate(old_model.policy.named_parameters())}
    o_state = old_model.policy.optimizer.state_dict()["state"]
    sd = model.policy.optimizer.state_dict()
    rows = [tuple(names.index(n) for n in pair.split("=")) for pair in copy]
    for i, (n, prm) in enumerate(model.policy.named_parameters()):
        st = o_state.get(o_idx.get(n, -1))
        if st is None:
            continue
        out = {"step": st["step"].clone()}
        for key in ("exp_avg", "exp_avg_sq"):
            v, t = st[key], torch.zeros_like(prm.data)
            t[tuple(slice(0, k) for k in v.shape)] = v
            if n.startswith("action_net.") and t.shape[0] > v.shape[0]:
                for to, frm in rows:
                    t[to] = t[frm]
            out[key] = t
        sd["state"][i] = out
    model.policy.optimizer.load_state_dict(sd)
    print(f"init_from: optimizer state carried for {len(sd['state'])} tensors")


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
    p.add_argument("--copy-rows", nargs="*", default=[], help="NEW=OLD action names: the new action starts as the old one")
    p.add_argument("--value-warmup", type=int, default=0, help="train only the value head this many steps first")
    p.add_argument("--eval-first", type=int, default=0, help="1: a [check] before any training (the starting policy in this env)")
    p.add_argument("--init", default="", help="start from an older checkpoint whose obs vector or action set was smaller (init_from)")
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--env", nargs="*", default=[], help="k=v for every env, train and eval (e.g. actions=184 no_ring=1)")
    p.add_argument("--target-kl", type=float, default=None, help="PPO stops an update's epochs past this KL")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=None, help="the learning rate falls linearly to this by --steps")
    p.add_argument("--ent-end", type=float, default=None, help="the entropy bonus falls linearly to this by --steps")
    p.add_argument("--live-seed", type=int, default=-1, help="the live world's seed: --live-pct of the train episodes there")
    p.add_argument("--live-pct", type=int, default=20)
    p.add_argument("--gear-pct", type=int, default=0, help="the share of train starts one step from a tool (rl/starts.py)")
    p.add_argument("--thin-pct", type=int, default=50, help="the share of train starts laid out like the live save")
    p.add_argument("--death-ends", type=int, default=1, help="0: as live, a death drops the pack and the episode goes on")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--days", type=float, default=0.75, help="episode length in days (from the morning): 0.75 is one night, 2.75 three")
    p.add_argument("--mask", type=int, default=1, help="1: MaskablePPO with the legal-action mask")
    a = p.parse_args()
    torch.set_num_threads(2)
    out = ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    ep = [f"ep_ticks={int(a.days * 12000)}", f"death_ends={a.death_ends}"] + a.env
    live = [f"live_seed={a.live_seed}", f"live_pct={a.live_pct}"] if a.live_seed >= 0 else []
    env = PipeVecEnv(a.envs, rng_seed=hash(a.run) % 1000, env_args=ep + live + [f"thin_pct={a.thin_pct}", f"gear_pct={a.gear_pct}"])
    ev = PipeVecEnv(a.eval_envs, env_args=ep)                                       # (held-out worlds as they are)
    ev.live_like = PipeVecEnv(a.eval_envs, env_args=ep + ["thin_pct=100"]) if a.thin_pct else None
    names_from(ev)
    bfile = ROOT / f"baselines_{a.days:g}d.json"
    base = json.loads(bfile.read_text()) if bfile.exists() else {}
    if "random_legal" not in base:
        base = baselines(ev, ev.spec["n_actions"])
    bfile.write_text(json.dumps(base, indent=1))
    algo = HybridMaskablePPO if a.mask else HybridPPO
    lr = a.lr if a.lr_end is None else (lambda left: a.lr_end + (a.lr - a.lr_end) * left)   # (left: 1 -> 0)
    if a.resume:
        model = load(algo, a.resume, env=env, custom_objects={"learning_rate": lr, "lr_schedule": lr if callable(lr) else (lambda _: lr), "ent_coef": a.ent,
                                                                           "target_kl": a.target_kl})
    else:
        model = algo("MultiInputPolicy", env, n_steps=512, batch_size=1024, n_epochs=4, gamma=a.gamma, gae_lambda=0.95,
                    learning_rate=lr, ent_coef=a.ent, clip_range=0.2, device="cpu", verbose=0, target_kl=a.target_kl,
                    policy_kwargs=POLICY_KW)
    if a.init:
        init_from(model, a.init, algo, a.copy_rows, env.spec["names"])
    cb = Check(a, out, ev, base)
    cb.log(f"[start] run={a.run} envs={a.envs} actions={env.spec['n_actions']} random={base['random']['score']} noop={base['noop']['score']}")
    try:
        if a.value_warmup:                        # (the critic first, policy frozen: after an env change it is stale)
            keep = [(n, p) for n, p in model.policy.named_parameters() if not n.startswith(("mlp_extractor.value_net", "value_net"))]
            for _, p in keep:
                p.requires_grad_(False)
            model.learn(total_timesteps=a.value_warmup, reset_num_timesteps=not a.resume)
            for _, p in keep:
                p.requires_grad_(True)
            cb.log(f"[warmup] value head trained {a.value_warmup} steps, policy frozen")
        model.learn(total_timesteps=a.steps, callback=cb, reset_num_timesteps=not a.resume and not a.value_warmup)
    finally:
        env.close()
        ev.close()
        if getattr(ev, "live_like", None) is not None:
            ev.live_like.close()


if __name__ == "__main__":
    main()
