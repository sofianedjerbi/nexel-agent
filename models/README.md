# models/rook.npz

The learned body of the shipped AI player, as numpy weights (`nexel_agent/rl/npolicy.py` loads it; no torch needed).
`python -m nexel_agent.rl.live --name Rook` uses it by default.

| | |
|---|---|
| Run | r27, best checkpoint (+600k steps of a resume from r25, which descends from r1 through r26) |
| Date | 2026-11-23 |
| Trained | PPO with the legal-action mask (sb3-contrib MaskablePPO), in the in-process sim (`rl/core.py`), 0.75-day episodes, death ends the episode, half the starts laid out like a lived-in save, a fifth on the live seed 12 |
| Held-out (48 lush worlds) | Crafter-style score 5.48, 3/48 deaths, 19.3 distinct crafts, 48 first strikes on neighbours (0.215 per minute spent within 2 tiles of one), shelter 0.90 |
| Live-like (24 held-out starts) | score 5.68, 2/24 deaths, 21.1 distinct crafts, 20 first strikes (0.155 per near-minute) |
| Baselines | random 1.21, legal random 2.24, noop 0.04 (same score) |
| Observation | `grid` 21x21 tile codes (70 chars), `height` 21x21, `vec` 476 floats (the code's `VEC` may be larger: the policy reads its first 476) |
| Actions | 184 (`rl/actions.py`; `ring`, the 185th, came after this run and is never chosen by it) |
| Size | 4.8 MB, sha256 `fa45531c5039cf3072245c5ac8fdf54a7879db46708239d80728b7bd3989649d` |

The words and the planner come from a local model, not from this file: serve Qwen3-30B-A3B-Instruct-2507 (4-bit) on
:8080 (`python -m mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080`), or the 4B on a
Mac with less than ~20 GB free. With no model server it plays on in silence.
