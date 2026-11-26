# models/rook.npz

The learned body of the shipped AI player, as numpy weights (`nexel_agent/rl/npolicy.py` loads it; no torch needed).
`python -m nexel_agent.rl.live --name Rook` uses it by default.

| | |
|---|---|
| Run | r50 best (+4.5M steps from r47: a first craft pays as deep as it is in the recipe tree, a spare end product costs a little; r47 from r46, r46 from r45; see rl_runs/LEDGER.md) |
| Date | 2026-11-26 |
| Trained | PPO with the legal-action mask (sb3-contrib MaskablePPO), in the in-process sim (`rl/core.py`): 0.75-day episodes, death ends the episode, food at the start between 8 and full, half the starts laid out like a lived-in save, a fifth on the live seed 12, 15% one step from a tool; goals paid when done, with a potential-based shaping of a recipe chain's progress (`rl/progress.py`) |
| Held-out (48 lush worlds, ship gate against r47, 4 passes, today's game with ores) | score 6.71 (r47 6.56), 0/48 deaths, closed rooms in 27% of episodes (r47 13%), build grade 0.43 of 6 (r47 0.22), tools 0.98 per episode (r47 0.82), goals done 0.62, repeat crafts 77% (r47 73%) |
| Live-like (24 held-out starts, same gate) | score 6.89 (r47 6.64), 0.25/24 deaths, rooms 26% (r47 15%), tools 0.94 (r47 0.80), failed actions 17.3% (r47 19.8%) |
| The bodies before it | r47 (2026-11-25): the r47 column above; r46, r37: rl_runs/LEDGER.md |
| Baselines | random 1.08, legal random 1.56, noop 0.04 (same score, same starts) |
| Observation | `grid` 21x21 tile codes (70 chars), `height` 21x21, `vec` 656 floats; tile chars, items, kinds and actions as frozen in `rl/vocab.py` |
| Actions | 187 (`rl/actions.py`: the 186 of `rl/vocab.py`, `pursue` included, then the game's new `craft hob`) |
| Size | 4.9 MB, sha256 `43cfe5cad3149968abdfee2977a4ac7ee86ed38a81705de9f5c4250e82183165` |

Known limits: spears, axes and pickaxes are still rare, about three crafts in four are still repeats of one already made, and it closes a room in few episodes (0 to 6% in these passes).

The words and the planner come from a local model, not from this file: serve Qwen3-30B-A3B-Instruct-2507 (4-bit) on
:8080 (`python -m mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080`), or the 4B on a
Mac with less than ~20 GB free. With no model server it plays on in silence.
