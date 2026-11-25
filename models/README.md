# models/rook.npz

The learned body of the shipped AI player, as numpy weights (`nexel_agent/rl/npolicy.py` loads it; no torch needed).
`python -m nexel_agent.rl.live --name Rook` uses it by default.

| | |
|---|---|
| Run | r47 best (+2.2M steps from r46: craft goals get time for their recipe chain, the spear among them; r46 from r45, the progress shaping; r45 from r43, the frozen vocabulary; r43 from r42, pursue; r42 from r37) |
| Date | 2026-11-25 |
| Trained | PPO with the legal-action mask (sb3-contrib MaskablePPO), in the in-process sim (`rl/core.py`): 0.75-day episodes, death ends the episode, food at the start between 8 and full, half the starts laid out like a lived-in save, a fifth on the live seed 12, 15% one step from a tool; goals paid when done, with a potential-based shaping of a recipe chain's progress (`rl/progress.py`) |
| Held-out (48 lush worlds, ship gate against r46, 4 passes, today's game) | score 6.55 (r46 5.81), 0/48 deaths (r46 0.5), 22.7 distinct crafts (r46 20.6), goals done 0.58 (r46 0.52), failed actions 21.4%, first strikes 0.004 per near-minute |
| Live-like (24 held-out starts, same gate) | score 6.65 (r46 5.89), 0.5/24 deaths (r46 0), 23.5 distinct crafts (r46 21.4), repeat crafts 73% (r46 75%), goals done 0.57 (r46 0.54) |
| The bodies before it | r46 (shipped earlier on 2026-11-25): the r46 column above; r37 (2026-11-24): lush score 5.52, 3.5/48 deaths, goals done 0.43; the rest in `rl_runs/LEDGER.md` |
| Baselines | random 1.08, legal random 1.56, noop 0.04 (same score, same starts) |
| Observation | `grid` 21x21 tile codes (70 chars), `height` 21x21, `vec` 656 floats; tile chars, items, kinds and actions as frozen in `rl/vocab.py` |
| Actions | 187 (`rl/actions.py`: the 186 of `rl/vocab.py`, `pursue` included, then the game's new `craft hob`) |
| Size | 4.9 MB, sha256 `8cef6c95b95da6cd3aea5dde44634e319c176e868894b656ca0a0dbc952bc5f6` |

Known limits: spears, axes and pickaxes are still rare, about three crafts in four are still repeats of one already made, and it closes a room in few episodes (0 to 6% in these passes).

The words and the planner come from a local model, not from this file: serve Qwen3-30B-A3B-Instruct-2507 (4-bit) on
:8080 (`python -m mlx_lm.server --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --port 8080`), or the 4B on a
Mac with less than ~20 GB free. With no model server it plays on in silence.
