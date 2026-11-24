# Training ledger (#7)

One entry per run: what it started from, what it changed (one thing), its config, and the gate's verdict
(`tools/ship_gate.py` appends those). Read this before starting the next run.

## The recipe (from r34–r41)

Resume from the shipped body with `--init` when the obs or the actions grew. lr 3e-5, ent 0.005, target_kl 0.01, a
75k-step value warm-up, a fifth of the train starts on the live seed 12, 15% one step from a tool, 0.75-day
episodes, death ends the episode. Change one thing per run.

## r42 (2026-11-24): pursue (#22)

- From: r37 at +600k (the shipped body), `--init` (the actions grew to 186), `--copy-rows pursue=approach`.
- Changed: one new action, `pursue` (rl/pursue.py): the goal's next step. Reward unchanged.
- The melee mask is the server's exact reach since #17, a tile more than r37 trained with: the `--eval-first`
  check is the step-0 baseline.
- First try stopped at the first check: pursue's walks failed (72% no_path in a probe), so fail% was 40.9. Fixed
  in 43be2b1 and restarted; the stopped run is rl_runs/r42-bad-pursue.out.
- Config: the recipe above, 3 envs, 2 h at most, a check every 150k, patience 4.
- Command: `PYTHONPATH=src:<nexel-worlds main>/src ~/minish-rl/.venv/bin/python -m nexel_agent.rl.train --run r42
  --init rl_runs/r37/at_600k.zip --copy-rows pursue=approach --envs 3 --eval-envs 3 --hours 2 --check 150000
  --patience 4 --lr 3e-5 --ent 0.005 --target-kl 0.01 --value-warmup 75000 --live-seed 12 --live-pct 20
  --gear-pct 15 --eval-first 1`

## gate 2026-11-25 00:55: rl_runs/r42/best.zip vs rl_runs/r37/at_600k.zip

| 4 passes | r37/at_600k.zip | r42/best.zip | |
|---|---|---|---|
| lush score | 5.526 ± 0.105 | 5.714 ± 0.244 | same |
| lush deaths | 4.000 ± 2.903 | 1.000 ± 1.298 | same |
| lush fs_per_near_min | 0.042 ± 0.023 | 0.030 ± 0.014 | same |
| lush crafts_distinct | 18.828 ± 1.300 | 19.725 ± 0.681 | same |
| lush repeat_crafts | 0.712 ± 0.008 | 0.720 ± 0.000 | same |
| lush fail_pct | 42.150 ± 4.322 | 28.675 ± 2.895 | better |
| lush tools_made | 0.510 ± 0.099 | 0.603 ± 0.053 | same |
| lush rooms | 0.015 ± 0.030 | 0.035 ± 0.030 | same |
| lush goals_done | 0.445 ± 0.023 | 0.474 ± 0.033 | same |
| lush goal_gather | 0.336 ± 0.038 | 0.363 ± 0.095 | same |
| lush goal_craft | 0.428 ± 0.118 | 0.458 ± 0.085 | same |
| lush goal_give | 0.848 ± 0.070 | 0.889 ± 0.235 | same |
| lush goal_ask | 0.735 ± 0.096 | 0.693 ± 0.163 | same |
| lush goal_stay | 0.542 ± 0.067 | 0.623 ± 0.248 | same |
| lush goal_help | 0.357 ± 0.013 | 0.395 ± 0.106 | same |
| lush goal_avoid | 0.263 ± 0.172 | 0.258 ± 0.146 | same |
| live-like score | 5.853 ± 0.341 | 5.749 ± 0.180 | same |
| live-like deaths | 1.250 ± 0.795 | 0.250 ± 0.795 | same |
| live-like fs_per_near_min | 0.044 ± 0.038 | 0.030 ± 0.010 | same |
| live-like crafts_distinct | 19.940 ± 0.151 | 20.470 ± 0.720 | same |
| live-like repeat_crafts | 0.705 ± 0.009 | 0.708 ± 0.015 | same |
| live-like fail_pct | 40.525 ± 10.656 | 28.025 ± 1.017 | better |
| live-like tools_made | 0.520 ± 0.110 | 0.573 ± 0.137 | same |
| live-like rooms | 0.010 ± 0.032 | 0.000 ± 0.000 | same |
| live-like goals_done | 0.423 ± 0.044 | 0.415 ± 0.045 | same |
| live-like goal_gather | 0.265 ± 0.105 | 0.364 ± 0.283 | same |
| live-like goal_craft | 0.347 ± 0.098 | 0.292 ± 0.143 | same |
| live-like goal_give | 0.749 ± 0.093 | 0.892 ± 0.134 | same |
| live-like goal_ask | 0.825 ± 0.376 | 0.658 ± 0.475 | same |
| live-like goal_stay | 0.607 ± 0.119 | 0.477 ± 0.190 | same |
| live-like goal_help | 0.370 ± 0.051 | 0.368 ± 0.062 | same |
| live-like goal_avoid | 0.334 ± 0.068 | 0.243 ± 0.099 | same |

verdict: keep the shipped body (targets: goals_done; 26 min)

### r42 result

The gate kept r37: goals done is "same" (0.474 ± 0.033 against 0.445 ± 0.023). Two findings:
- The shipped r37 fails 42% of its actions in today's game (it trained at about 20%): the game changed under it, the
  reach mask included (#17). r42 is better there, at 28% (CI clear of r37's), with every other row the same.
- r42 trained on the pursue that could not get a tool started (dc43c36 fixed it).

## r43 (2026-11-25): pursue that can make tools

- From: r42 best (at +225k, already playing today's game), `--init` (same shapes).
- Changed: only pursue's craft step (dc43c36): any input it lacks that is at hand, gathered before it is made.
- Config: the recipe, 3 envs, 2 h at most, `nice -n 10`.
