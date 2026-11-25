# models/rook.npz

The learned body of the shipped AI player, as numpy weights (`nexel_agent/rl/npolicy.py` loads it; no torch needed).
`python -m nexel_agent.rl.live --name Rook` uses it by default.

| | |
|---|---|
| Run | r46 best (from r45 best, the progress shaping of craft and gather goals; r45 came from r43 with the vocabulary frozen; r43 from r42, pursue; r42 from r37) |
| Date | 2026-11-25 |
| Trained | PPO with the legal-action mask (sb3-contrib MaskablePPO), in the in-process sim (`rl/core.py`): 0.75-day episodes, death ends the episode, food at the start between 8 and full, half the starts laid out like a lived-in save, a fifth on the live seed 12, 15% one step from a tool; goals paid when done, with a potential-based shaping of a recipe chain's progress (`rl/progress.py`) |
| Held-out (48 lush worlds, ship gate, 4 passes, today's game) | score 5.71 (r37 5.52), 0.5/48 deaths (r37 3.5), goals done 0.537 (r37 0.430), tools or weapons made 0.88 per episode (r37 0.62), failed actions 20.9% (r37 39.6%), first strikes 0.003 per near-minute (r37 0.038) |
| Live-like (24 held-out starts, same gate) | score 5.89 (r37 5.56), 0.25/24 deaths (r37 2.25), goals done 0.506 (r37 0.442), tools 0.89 (r37 0.51), failed actions 20.8% (r37 39.3%), first strikes 0.010 (r37 0.056) |
| The bodies before it | r37 at +600k (shipped 2026-11-24): the r37 column above; earlier ones in `rl_runs/LEDGER.md` |
| Baselines | random 1.08, legal random 1.56, noop 0.04 (same score, same starts) |
| Observation | `grid` 21x21 tile codes (70 chars), `height` 21x21, `vec` 656 floats; tile chars, items, kinds and actions as frozen in `rl/vocab.py` |
| Actions | 187 (`rl/actions.py`: the 186 of `rl/vocab.py`, `pursue` included, then the game's new `craft hob`) |
| Size | 4.9 MB, sha256 `12901ae988fea5b28f811b3a6a075de8ec67fced3824657fe2c1f614a7017967` |

Known limits: spears, axes and pickaxes are still rare, about three crafts in four are still repeats of one already made, and it closes a room in few episodes (0 to 6% in these passes).

The words and the planner come from a local model, not from this file: serve Qwen3-30B-A3B-Instruct-2507 (4-bit) on
:8080 (`python -m mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080`), or the 4B on a
Mac with less than ~20 GB free. With no model server it plays on in silence.
