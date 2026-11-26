# models/rook.npz

The learned body of the shipped AI player, as numpy weights (`nexel_agent/rl/npolicy.py` loads it; no torch needed).
`python -m nexel_agent.rl.live --name Rook` uses it by default.

| | |
|---|---|
| Run | r63 last (+1.5M from r62; r51-r63 taught the build verb: one pick lays a hut, home-kit and home-begun starts, a potential on the judge's grade and the pieces laid; on nexel-worlds 4f94113c, ores and tool tiers in the vocabulary) |
| Date | 2026-11-26 |
| Trained | PPO with the legal-action mask (sb3-contrib MaskablePPO), in the in-process sim (`rl/core.py`): 0.75-day episodes, death ends the episode, food at the start between 8 and full, half the starts laid out like a lived-in save, a fifth on the live seed 12, 15% one step from a tool; goals paid when done, with a potential-based shaping of a recipe chain's progress (`rl/progress.py`) |
| Held-out (48 lush worlds, ship gate against r50, 4 passes, today's game) | build grade 0.80 of 6 (r50 0.50), closed rooms 43% (r50 30%), tools 1.40 per episode (r50 1.04), score 6.07 (r50 6.14, same), 0.5/48 deaths, first strikes 0.001 per near-minute |
| Live-like (24 held-out starts, same gate) | build grade 0.73 (r50 0.64), rooms 42% (r50 38%), tools 1.33 (r50 0.96), repeat crafts 75% (r50 77%), failed actions 18.5% (r50 16.2%, worse) |
| The bodies before it | r50 (2026-11-26): the r50 column above; r47, r46, r37: rl_runs/LEDGER.md |
| Baselines | random 1.08, legal random 1.56, noop 0.04 (same score, same starts) |
| Observation | `grid` 21x21 tile codes (70 chars), `height` 21x21, `vec` 688 floats; tile chars, items, kinds and actions as frozen in `rl/vocab.py` (ITEMS_V2: the ores, ingots and tool tiers) |
| Actions | 200 (`rl/vocab.py`, `pursue` and `build` included) |
| Size | 4.9 MB, sha256 `b5250737f25e0d8ad9641ec461fee8a69608664b50de58635bdc0392fab1f6b6` |

Known limits: spears, axes and pickaxes are still rare, about three crafts in four are still repeats of one already made, and it closes a room in few episodes (0 to 6% in these passes).

The words and the planner come from a local model, not from this file: serve Qwen3-30B-A3B-Instruct-2507 (4-bit) on
:8080 (`python -m mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080`), or the 4B on a
Mac with less than ~20 GB free. With no model server it plays on in silence.
