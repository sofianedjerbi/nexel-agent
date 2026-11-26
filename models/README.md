# models/rook.npz

The learned body of the shipped AI player, as numpy weights (`nexel_agent/rl/npolicy.py` loads it; no torch needed).
`python -m nexel_agent.rl.live --name Rook` uses it by default.

| | |
|---|---|
| Run | r64 best (from r63: the sim's neighbours a mixture, 40% quiet like the save's people, 3% rough; a home with pieces standing kept) |
| Date | 2026-11-26 |
| Trained | PPO with the legal-action mask (sb3-contrib MaskablePPO), in the in-process sim (`rl/core.py`): 0.75-day episodes, death ends the episode, food at the start between 8 and full, half the starts laid out like a lived-in save, a fifth on the live seed 12, 15% one step from a tool; goals paid when done, with a potential-based shaping of a recipe chain's progress (`rl/progress.py`) |
| Held-out (48 lush worlds, ship gate against r63, 4 passes) | score 6.37 (r63 5.95), closed rooms 55% (r63 36%), build grade 1.01 of 6 (r63 0.61), tools 1.56, 0.25/48 deaths, first strikes 0.001 per near-minute, failed actions 18.2% |
| Live-like (24 held-out starts, same gate) | score 6.62 (r63 6.08), rooms 57% (r63 29%), build grade 1.07 (r63 0.49), tools 1.49, 0/24 deaths, failed actions 17.2% |
| The bodies before it | r63 (2026-11-26): the r63 column above; r50, r47, r46, r37: rl_runs/LEDGER.md |
| Baselines | random 1.08, legal random 1.56, noop 0.04 (same score, same starts) |
| Observation | `grid` 21x21 tile codes (70 chars), `height` 21x21, `vec` 688 floats; tile chars, items, kinds and actions as frozen in `rl/vocab.py` (ITEMS_V2: the ores, ingots and tool tiers) |
| Actions | 200 (`rl/vocab.py`, `pursue` and `build` included) |
| Size | 4.9 MB, sha256 `9a9f39f26f56392bd4bf0cafe69ea9a58c5843e62e17d6e42db36fb8a05b8d77` |

Known limits: spears, axes and pickaxes are still rare, about three crafts in four are still repeats of one already made, and it closes a room in few episodes (0 to 6% in these passes).

The words and the planner come from a local model, not from this file: serve Qwen3-30B-A3B-Instruct-2507 (4-bit) on
:8080 (`python -m mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080`), or the 4B on a
Mac with less than ~20 GB free. With no model server it plays on in silence.
