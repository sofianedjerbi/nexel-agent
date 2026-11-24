# models/rook.npz

The learned body of the shipped AI player, as numpy weights (`nexel_agent/rl/npolicy.py` loads it; no torch needed).
`python -m nexel_agent.rl.live --name Rook` uses it by default.

| | |
|---|---|
| Run | r34 at +600k steps (init from r27 with the optimizer state carried over; r27 descends from r1) |
| Date | 2026-11-24 |
| Trained | PPO with the legal-action mask (sb3-contrib MaskablePPO), in the in-process sim (`rl/core.py`): 0.75-day episodes, death ends the episode, food at the start between 8 and full, half the starts laid out like a lived-in save, a fifth on the live seed 12 |
| Held-out (48 lush worlds, mean of 2 passes) | Crafter-style score 5.45, 4.0/48 deaths, 19.4 distinct crafts, 26.5 first strikes on neighbours (0.103 per minute spent within 2 tiles of one), shelter 0.89 |
| Live-like (24 held-out starts, mean of 2 passes) | score 5.64, 1.5/24 deaths, 19.7 distinct crafts, 19.5 first strikes (0.170 per near-minute) |
| The body before it (r27), same worlds and passes | held-out 5.38, 5.5 deaths, 0.239 first strikes per near-minute; live-like 5.58, 4.0 deaths, 0.242 |
| Baselines | random 1.08, legal random 1.56, noop 0.04 (same score, same starts) |
| Observation | `grid` 21x21 tile codes (70 chars), `height` 21x21, `vec` 477 floats |
| Actions | 185 (`rl/actions.py`, `ring` included) |
| Size | 4.8 MB, sha256 `a673f973af235c2a920a730086b9c2118d8fff2474e3245b0dd4efff23607be6` |

Known limits: it crafts no weapons or tools, and closes a room in few episodes (none in these four passes; 4% in the
training checks).

The words and the planner come from a local model, not from this file: serve Qwen3-30B-A3B-Instruct-2507 (4-bit) on
:8080 (`python -m mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080`), or the 4B on a
Mac with less than ~20 GB free. With no model server it plays on in silence.
