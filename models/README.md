# models/rook.npz

The learned body of the shipped AI player, as numpy weights (`nexel_agent/rl/npolicy.py` loads it; no torch needed).
`python -m nexel_agent.rl.live --name Rook` uses it by default.

| | |
|---|---|
| Run | r37 at +600k steps (from r35 at +300k, the vector grown by the done bits, a 75k-step value warmup first; r35 came from r34 at +900k; r27 descends from r1) |
| Date | 2026-11-24 |
| Trained | PPO with the legal-action mask (sb3-contrib MaskablePPO), in the in-process sim (`rl/core.py`): 0.75-day episodes, death ends the episode, food at the start between 8 and full, half the starts laid out like a lived-in save, a fifth on the live seed 12, 15% one step from a tool |
| Held-out (48 lush worlds, mean of 2 passes) | Crafter-style score 5.83, 1.0/48 deaths, 20.5 distinct crafts, 10.0 first strikes on neighbours (0.036 per minute spent within 2 tiles of one), shelter 0.96, a room in 4% of episodes |
| Live-like (24 held-out starts, mean of 2 passes) | score 6.12, 0.0/24 deaths, 22.1 distinct crafts, 6.5 first strikes (0.049 per near-minute), a room in 2% of episodes |
| The bodies before it, same worlds and passes | r35 at +300k: held-out 5.83, 1.0 deaths, 0.061 first strikes per near-minute; live-like 5.89, 1.5 deaths, 0.085. r34 at +900k: held-out 5.63, 1.0 deaths, 0.067 first strikes per near-minute; live-like 5.65, 1.5 deaths, 0.097. r34 at +600k: held-out 5.45, 4.0 deaths, 0.103 first strikes per near-minute; live-like 5.64, 1.5 deaths, 0.170. r27: held-out 5.38, 5.5 deaths, 0.239; live-like 5.58, 4.0 deaths, 0.242 |
| Baselines | random 1.08, legal random 1.56, noop 0.04 (same score, same starts) |
| Observation | `grid` 21x21 tile codes (70 chars), `height` 21x21, `vec` 654 floats (the last 177: what it has already done this life, `Ledger.done`) |
| Actions | 185 (`rl/actions.py`, `ring` included) |
| Size | 4.9 MB, sha256 `7a1ddaf97d323bee51b524eccc0dcce4f2c46fab513f21428b0e511a517d7a1d` |

Known limits: it crafts no weapons or tools, about three crafts in four are still repeats of one already made, and it closes a room in few episodes (0 to 6% in these passes).

The words and the planner come from a local model, not from this file: serve Qwen3-30B-A3B-Instruct-2507 (4-bit) on
:8080 (`python -m mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080`), or the 4B on a
Mac with less than ~20 GB free. With no model server it plays on in silence.
