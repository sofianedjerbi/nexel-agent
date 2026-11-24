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
