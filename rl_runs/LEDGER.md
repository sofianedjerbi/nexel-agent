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

## r44

- From: r43 best, `--init` (same shapes).
- Changed: pursue finds fibre in a picked berry bush and leaves ore it cannot break (6431d2a); the masks read the
  game's needs_tool, untouchable, ENCLOSE (21028ee, 9fef5f9). r43 made 0 tools: 5 of 7 worlds had no fibre source
  pursue knew of within 60 tiles. Probe, pursue alone: tools 3/18 -> 11/18.
- Config: as r43 (3 envs, 2 h at most, `nice -n 10`). Target: tools (and goals_done).

## gate 2026-11-25 04:06: rl_runs/r43/best.zip vs rl_runs/r37/at_600k.zip

| 4 passes | r37/at_600k.zip | r43/best.zip | |
|---|---|---|---|
| lush score | 5.526 ± 0.105 | 5.877 ± 0.243 | better |
| lush deaths | 4.000 ± 2.903 | 0.750 ± 0.795 | better |
| lush fs_per_near_min | 0.042 ± 0.023 | 0.020 ± 0.007 | same |
| lush crafts_distinct | 18.828 ± 1.300 | 20.483 ± 0.523 | better |
| lush repeat_crafts | 0.712 ± 0.008 | 0.742 ± 0.008 | worse |
| lush fail_pct | 42.150 ± 4.322 | 23.500 ± 2.567 | better |
| lush tools_made | 0.510 ± 0.099 | 0.688 ± 0.157 | same |
| lush rooms | 0.015 ± 0.030 | 0.035 ± 0.016 | same |
| lush goals_done | 0.445 ± 0.023 | 0.513 ± 0.079 | same |
| lush goal_gather | 0.336 ± 0.038 | 0.522 ± 0.102 | better |
| lush goal_craft | 0.428 ± 0.118 | 0.617 ± 0.121 | better |
| lush goal_give | 0.848 ± 0.070 | 0.920 ± 0.155 | same |
| lush goal_ask | 0.735 ± 0.096 | 0.687 ± 0.201 | same |
| lush goal_stay | 0.542 ± 0.067 | 0.599 ± 0.046 | same |
| lush goal_help | 0.357 ± 0.013 | 0.392 ± 0.075 | same |
| lush goal_avoid | 0.263 ± 0.172 | 0.259 ± 0.224 | same |
| live-like score | 5.853 ± 0.341 | 5.913 ± 0.109 | same |
| live-like deaths | 1.250 ± 0.795 | 0.250 ± 0.795 | same |
| live-like fs_per_near_min | 0.044 ± 0.038 | 0.016 ± 0.026 | same |
| live-like crafts_distinct | 19.940 ± 0.151 | 20.635 ± 0.378 | better |
| live-like repeat_crafts | 0.705 ± 0.009 | 0.742 ± 0.008 | worse |
| live-like fail_pct | 40.525 ± 10.656 | 19.550 ± 1.715 | better |
| live-like tools_made | 0.520 ± 0.110 | 0.742 ± 0.149 | better |
| live-like rooms | 0.010 ± 0.032 | 0.010 ± 0.032 | same |
| live-like goals_done | 0.423 ± 0.044 | 0.488 ± 0.078 | same |
| live-like goal_gather | 0.265 ± 0.105 | 0.431 ± 0.289 | same |
| live-like goal_craft | 0.347 ± 0.098 | 0.471 ± 0.079 | same |
| live-like goal_give | 0.749 ± 0.093 | 0.979 ± 0.066 | better |
| live-like goal_ask | 0.825 ± 0.376 | 0.708 ± 0.328 | same |
| live-like goal_stay | 0.607 ± 0.119 | 0.512 ± 0.289 | same |
| live-like goal_help | 0.370 ± 0.051 | 0.430 ± 0.075 | same |
| live-like goal_avoid | 0.334 ± 0.068 | 0.226 ± 0.166 | same |

verdict: keep the shipped body (targets: goals_done; 20 min)
- **Result: void.** The game's new creatures and hob shifted the encoder's ids (30 tile chars, 66 actions): r44 began
  from r43 with a scrambled view and crossed action rows (deaths 14-25/48, score about 4.3). Renamed r44-scrambled.
  Fixed by the frozen vocabulary (5aa0548).

## r45

- As r44 (from r43 best, the fibre pursue), with the body's vocabulary frozen as it learned it (rl/vocab.py). Today's
  game (nexel-worlds b5671f6): new creatures read as unknown chars, the hob craft is a new last action.

- **Result (r45):** sane again (deaths 0-2/48, best score 5.78), tools still ~0 (one check 0.02 spears). Not gated: its
  target (tools) is not met, so it cannot ship.

## r46

- From: r45's best (already on today's game), `--init`.
- Changed: the progress shaping of craft and gather goals (rl/progress.py): potential-based, 0.05 a step of the
  recipe tree, paid back when the goal ends. r43 picked pursue 7% of the times it was legal under a tool goal.
- Target: tools_made (and goals_done); the guards as ever.

## gate 2026-11-25 10:38: rl_runs/r46/best.zip vs rl_runs/r37/at_600k.zip

| 4 passes | r37/at_600k.zip | r46/best.zip | |
|---|---|---|---|
| lush score | 5.518 ± 0.306 | 5.712 ± 0.151 | same |
| lush deaths | 3.500 ± 4.590 | 0.500 ± 0.918 | same |
| lush fs_per_near_min | 0.038 ± 0.031 | 0.003 ± 0.006 | better |
| lush crafts_distinct | 19.200 ± 0.788 | 20.553 ± 0.356 | better |
| lush repeat_crafts | 0.718 ± 0.015 | 0.738 ± 0.008 | worse |
| lush fail_pct | 39.625 ± 5.691 | 20.875 ± 0.457 | better |
| lush tools_made | 0.623 ± 0.158 | 0.883 ± 0.038 | better |
| lush rooms | 0.015 ± 0.016 | 0.092 ± 0.105 | same |
| lush goals_done | 0.430 ± 0.037 | 0.537 ± 0.035 | better |
| lush goal_gather | 0.361 ± 0.091 | 0.582 ± 0.063 | better |
| lush goal_craft | 0.314 ± 0.067 | 0.684 ± 0.053 | better |
| lush goal_give | 0.732 ± 0.219 | 0.909 ± 0.055 | same |
| lush goal_ask | 0.761 ± 0.072 | 0.789 ± 0.130 | same |
| lush goal_stay | 0.533 ± 0.245 | 0.677 ± 0.171 | same |
| lush goal_help | 0.387 ± 0.017 | 0.373 ± 0.057 | same |
| lush goal_avoid | 0.255 ± 0.069 | 0.231 ± 0.061 | same |
| live-like score | 5.562 ± 0.293 | 5.890 ± 0.349 | same |
| live-like deaths | 2.250 ± 1.522 | 0.250 ± 0.795 | better |
| live-like fs_per_near_min | 0.056 ± 0.032 | 0.010 ± 0.006 | better |
| live-like crafts_distinct | 19.605 ± 1.010 | 21.125 ± 0.354 | better |
| live-like repeat_crafts | 0.695 ± 0.016 | 0.750 ± 0.013 | worse |
| live-like fail_pct | 39.325 ± 13.856 | 20.750 ± 2.539 | better |
| live-like tools_made | 0.510 ± 0.109 | 0.887 ± 0.068 | better |
| live-like rooms | 0.020 ± 0.037 | 0.073 ± 0.103 | same |
| live-like goals_done | 0.442 ± 0.054 | 0.506 ± 0.042 | same |
| live-like goal_gather | 0.259 ± 0.124 | 0.505 ± 0.200 | better |
| live-like goal_craft | 0.229 ± 0.076 | 0.536 ± 0.125 | better |
| live-like goal_give | 0.936 ± 0.122 | 0.877 ± 0.201 | same |
| live-like goal_ask | 0.525 ± 0.654 | 0.732 ± 0.494 | same |
| live-like goal_stay | 0.584 ± 0.255 | 0.671 ± 0.059 | same |
| live-like goal_help | 0.425 ± 0.088 | 0.397 ± 0.077 | same |
| live-like goal_avoid | 0.354 ± 0.169 | 0.284 ± 0.076 | same |

verdict: keep the shipped body (targets: goals_done; 20 min)
- **Result (r46):** the strongest body yet against r37 on today's game: goals_done better on lush (0.537 vs 0.430),
  same on live-like (0.506 vs 0.442); tools_made, fail%, first strikes better on both; repeat crafts worse; no guard
  worse. Verdict by the rule: keep r37 (the target must be better on both sets). Shipping r46 is the owner's call.
- **Shipped (owner's call: the smartest body wins):** r46 is models/rook.npz (f1af1e4).

## r47

- From: r46 best (shipped), `--init`.
- Changed: a craft goal gets 20 s more per step of its recipe tree (a tool: ~420 s, was 180 s); the spear is among
  the sim's craft goals (ee531ff).
- Config: as r46, but 3 h at most and patience 6.

## r48

- From: r47 best (shipped), `--init`, a 75k value warmup (the craft rewards grew).
- Changed: a first craft pays as deep as it is, 1 to 4 (a spear 2.75, an axe 3; was 1 for all).

## gate 2026-11-25 14:29: rl_runs/r47/best.zip vs rl_runs/r46/best.zip

| 4 passes | r46/best.zip | r47/best.zip | |
|---|---|---|---|
| lush score | 5.805 ± 0.109 | 6.546 ± 0.203 | better |
| lush deaths | 0.500 ± 0.918 | 0.000 ± 0.000 | same |
| lush fs_per_near_min | 0.003 ± 0.005 | 0.004 ± 0.007 | same |
| lush crafts_distinct | 20.615 ± 0.149 | 22.732 ± 0.965 | better |
| lush repeat_crafts | 0.740 ± 0.000 | 0.730 ± 0.013 | same |
| lush fail_pct | 21.225 ± 1.482 | 21.425 ± 3.329 | same |
| lush tools_made | 0.870 ± 0.079 | 0.823 ± 0.166 | same |
| lush rooms | 0.083 ± 0.099 | 0.130 ± 0.075 | same |
| lush goals_done | 0.516 ± 0.031 | 0.580 ± 0.057 | same |
| lush goal_gather | 0.580 ± 0.037 | 0.642 ± 0.128 | same |
| lush goal_craft | 0.545 ± 0.111 | 0.681 ± 0.084 | same |
| lush goal_give | 0.971 ± 0.053 | 1.000 ± 0.000 | same |
| lush goal_ask | 0.830 ± 0.033 | 0.894 ± 0.072 | same |
| lush goal_stay | 0.639 ± 0.118 | 0.702 ± 0.105 | same |
| lush goal_help | 0.356 ± 0.042 | 0.394 ± 0.085 | same |
| lush goal_avoid | 0.235 ± 0.116 | 0.289 ± 0.071 | same |
| live-like score | 5.889 ± 0.372 | 6.652 ± 0.377 | better |
| live-like deaths | 0.000 ± 0.000 | 0.500 ± 0.918 | same |
| live-like fs_per_near_min | 0.002 ± 0.006 | 0.003 ± 0.006 | same |
| live-like crafts_distinct | 21.355 ± 0.621 | 23.532 ± 0.600 | better |
| live-like repeat_crafts | 0.748 ± 0.008 | 0.728 ± 0.015 | better |
| live-like fail_pct | 20.225 ± 1.050 | 19.600 ± 2.267 | same |
| live-like tools_made | 0.878 ± 0.097 | 0.823 ± 0.116 | same |
| live-like rooms | 0.080 ± 0.052 | 0.168 ± 0.145 | same |
| live-like goals_done | 0.535 ± 0.083 | 0.569 ± 0.055 | same |
| live-like goal_gather | 0.521 ± 0.226 | 0.549 ± 0.188 | same |
| live-like goal_craft | 0.666 ± 0.038 | 0.784 ± 0.049 | better |
| live-like goal_give | 0.917 ± 0.098 | 0.946 ± 0.059 | same |
| live-like goal_ask | 0.775 ± 0.285 | 0.762 ± 0.115 | same |
| live-like goal_stay | 0.666 ± 0.125 | 0.757 ± 0.106 | same |
| live-like goal_help | 0.392 ± 0.129 | 0.365 ± 0.127 | same |
| live-like goal_avoid | 0.319 ± 0.096 | 0.325 ± 0.210 | same |

verdict: SHIP (targets: score; 18 min)
- **Result (r47):** best 6.83 at +2.2M; gate against r46 (4 passes): score better on both sets (lush 5.81 -> 6.55,
  live-like 5.89 -> 6.65), distinct crafts better, nothing worse. **Shipped.**

## gate 2026-11-25 16:03: rl_runs/r48/best.zip vs rl_runs/r47/best.zip

| 4 passes | r47/best.zip | r48/best.zip | |
|---|---|---|---|
| lush score | 6.546 ± 0.203 | 6.622 ± 0.179 | same |
| lush deaths | 0.000 ± 0.000 | 0.750 ± 1.522 | same |
| lush fs_per_near_min | 0.004 ± 0.007 | 0.004 ± 0.005 | same |
| lush crafts_distinct | 22.732 ± 0.965 | 23.100 ± 0.259 | same |
| lush repeat_crafts | 0.730 ± 0.013 | 0.765 ± 0.009 | worse |
| lush fail_pct | 21.425 ± 3.329 | 18.725 ± 1.544 | same |
| lush tools_made | 0.823 ± 0.166 | 0.938 ± 0.053 | same |
| lush rooms | 0.130 ± 0.075 | 0.170 ± 0.112 | same |
| lush goals_done | 0.580 ± 0.057 | 0.595 ± 0.028 | same |
| lush goal_gather | 0.642 ± 0.128 | 0.620 ± 0.099 | same |
| lush goal_craft | 0.681 ± 0.084 | 0.683 ± 0.164 | same |
| lush goal_give | 1.000 ± 0.000 | 0.974 ± 0.049 | same |
| lush goal_ask | 0.894 ± 0.072 | 0.839 ± 0.111 | same |
| lush goal_stay | 0.702 ± 0.105 | 0.734 ± 0.139 | same |
| lush goal_help | 0.394 ± 0.085 | 0.430 ± 0.040 | same |
| lush goal_avoid | 0.289 ± 0.071 | 0.261 ± 0.214 | same |
| live-like score | 6.652 ± 0.377 | 6.832 ± 0.217 | same |
| live-like deaths | 0.500 ± 0.918 | 0.000 ± 0.000 | same |
| live-like fs_per_near_min | 0.003 ± 0.006 | 0.002 ± 0.005 | same |
| live-like crafts_distinct | 23.532 ± 0.600 | 24.385 ± 0.788 | same |
| live-like repeat_crafts | 0.728 ± 0.015 | 0.768 ± 0.015 | worse |
| live-like fail_pct | 19.600 ± 2.267 | 19.750 ± 6.163 | same |
| live-like tools_made | 0.823 ± 0.116 | 1.000 ± 0.073 | better |
| live-like rooms | 0.168 ± 0.145 | 0.122 ± 0.097 | same |
| live-like goals_done | 0.569 ± 0.055 | 0.589 ± 0.073 | same |
| live-like goal_gather | 0.549 ± 0.188 | 0.827 ± 0.057 | better |
| live-like goal_craft | 0.784 ± 0.049 | 0.819 ± 0.055 | same |
| live-like goal_give | 0.946 ± 0.059 | 0.939 ± 0.065 | same |
| live-like goal_ask | 0.762 ± 0.115 | 0.806 ± 0.082 | same |
| live-like goal_stay | 0.757 ± 0.106 | 0.698 ± 0.033 | same |
| live-like goal_help | 0.365 ± 0.127 | 0.376 ± 0.103 | same |
| live-like goal_avoid | 0.325 ± 0.210 | 0.266 ± 0.130 | same |

verdict: keep the shipped body (targets: tools_made; 19 min)
- **Result (r48):** gate against r47 (target tools_made): tools better on live-like (0.82 -> 1.00), same on lush; score,
  goals, deaths same; repeat crafts worse (0.73 -> 0.77). Not clearly smarter: r47 stays.

## r49

- From: r48 best, `--init`.
- Changed: another craft of a thing already unused in the pack costs 0.05 (not materials, not room pieces).
  In 3 r47 episodes: 541 crafts, 468 repeats (33 mailboxes, 31 ladders, 30 chairs, 30 troughs, 26 extra benches).
- Target: repeat crafts down with tools and score no worse.

## gate 2026-11-25 17:43: rl_runs/r49/best.zip vs rl_runs/r47/best.zip

| 4 passes | r47/best.zip | r49/best.zip | |
|---|---|---|---|
| lush score | 6.546 ± 0.203 | 6.500 ± 0.165 | same |
| lush deaths | 0.000 ± 0.000 | 0.750 ± 0.795 | same |
| lush fs_per_near_min | 0.004 ± 0.007 | 0.005 ± 0.007 | same |
| lush crafts_distinct | 22.732 ± 0.965 | 23.157 ± 0.820 | same |
| lush repeat_crafts | 0.730 ± 0.013 | 0.775 ± 0.009 | worse |
| lush fail_pct | 21.425 ± 3.329 | 21.350 ± 1.193 | same |
| lush tools_made | 0.823 ± 0.166 | 0.927 ± 0.105 | same |
| lush rooms | 0.130 ± 0.075 | 0.260 ± 0.134 | same |
| lush goals_done | 0.580 ± 0.057 | 0.570 ± 0.049 | same |
| lush goal_gather | 0.642 ± 0.128 | 0.624 ± 0.089 | same |
| lush goal_craft | 0.681 ± 0.084 | 0.695 ± 0.143 | same |
| lush goal_give | 1.000 ± 0.000 | 1.000 ± 0.000 | same |
| lush goal_ask | 0.894 ± 0.072 | 0.926 ± 0.100 | same |
| lush goal_stay | 0.702 ± 0.105 | 0.675 ± 0.103 | same |
| lush goal_help | 0.394 ± 0.085 | 0.397 ± 0.050 | same |
| lush goal_avoid | 0.289 ± 0.071 | 0.214 ± 0.130 | same |
| live-like score | 6.652 ± 0.377 | 6.796 ± 0.223 | same |
| live-like deaths | 0.500 ± 0.918 | 0.500 ± 1.590 | same |
| live-like fs_per_near_min | 0.003 ± 0.006 | 0.003 ± 0.006 | same |
| live-like crafts_distinct | 23.532 ± 0.600 | 23.010 ± 1.404 | same |
| live-like repeat_crafts | 0.728 ± 0.015 | 0.762 ± 0.015 | worse |
| live-like fail_pct | 19.600 ± 2.267 | 19.900 ± 1.030 | same |
| live-like tools_made | 0.823 ± 0.116 | 0.885 ± 0.116 | same |
| live-like rooms | 0.168 ± 0.145 | 0.220 ± 0.061 | same |
| live-like goals_done | 0.569 ± 0.055 | 0.603 ± 0.072 | same |
| live-like goal_gather | 0.549 ± 0.188 | 0.785 ± 0.109 | better |
| live-like goal_craft | 0.784 ± 0.049 | 0.821 ± 0.061 | same |
| live-like goal_give | 0.946 ± 0.059 | 1.000 ± 0.000 | same |
| live-like goal_ask | 0.762 ± 0.115 | 0.916 ± 0.153 | same |
| live-like goal_stay | 0.757 ± 0.106 | 0.730 ± 0.186 | same |
| live-like goal_help | 0.365 ± 0.127 | 0.373 ± 0.104 | same |
| live-like goal_avoid | 0.325 ± 0.210 | 0.260 ± 0.101 | same |

verdict: keep the shipped body (targets: score; 18 min)
- **Result (r49):** gate against r47: all same, repeat crafts worse (0.73 -> 0.77). The junk moved, not gone: in 3
  episodes, mailboxes/ladders/troughs fell out of the top, stone (179) and stone walls (71, which the wall stock pays)
  took their place. Not shipped. The repeat-crafts KPI counts materials and walls too: a crude measure of waste.

## r50

- From: r47 best (shipped), `--init`. With both craft rewards (depth pay, the spare-craft cost), from the shipped body
  rather than r48's; 4 h at most, lr 2e-5. Queued behind another agent's job.

## r51

- From: r50 best (shipped), `--init` (one new action: `build`, rl/build.py; a 75k value warmup).
- Changed: the `build` verb (a 3x3 hut, laid step by step) and the build reward (a potential on the judge's grade,
  0.25 a check: rl/judge.py). The game pin moves to main (roofs in the observation, ores and tool tiers).
- Target: builds_5 (best build passes 5 of 6 checks) up from about 0, score and deaths no worse.

## gate 2026-11-26 01:20: rl_runs/r50/best.zip vs rl_runs/r47/best.zip

| 4 passes | r47/best.zip | r50/best.zip | |
|---|---|---|---|
| lush score | 6.556 ± 0.168 | 6.708 ± 0.477 | same |
| lush deaths | 0.000 ± 0.000 | 0.000 ± 0.000 | same |
| lush fs_per_near_min | 0.004 ± 0.007 | 0.004 ± 0.010 | same |
| lush crafts_distinct | 22.703 ± 0.879 | 24.345 ± 0.633 | better |
| lush repeat_crafts | 0.732 ± 0.008 | 0.770 ± 0.000 | worse |
| lush fail_pct | 21.400 ± 3.140 | 20.775 ± 4.110 | same |
| lush tools_made | 0.823 ± 0.166 | 0.980 ± 0.094 | same |
| lush rooms | 0.130 ± 0.075 | 0.270 ± 0.110 | better |
| lush build_grade | 0.215 ± 0.105 | 0.432 ± 0.201 | same |
| lush builds_5 | 0.000 ± 0.000 | 0.000 ± 0.000 | same |
| lush goals_done | 0.579 ± 0.038 | 0.621 ± 0.044 | same |
| lush goal_gather | 0.640 ± 0.125 | 0.661 ± 0.045 | same |
| lush goal_craft | 0.690 ± 0.091 | 0.789 ± 0.090 | same |
| lush goal_give | 1.000 ± 0.000 | 0.986 ± 0.045 | same |
| lush goal_ask | 0.872 ± 0.088 | 0.888 ± 0.034 | same |
| lush goal_stay | 0.710 ± 0.132 | 0.667 ± 0.112 | same |
| lush goal_help | 0.385 ± 0.063 | 0.446 ± 0.053 | same |
| lush goal_avoid | 0.295 ± 0.064 | 0.323 ± 0.208 | same |
| live-like score | 6.641 ± 0.355 | 6.889 ± 0.215 | same |
| live-like deaths | 0.500 ± 0.918 | 0.250 ± 0.795 | same |
| live-like fs_per_near_min | 0.002 ± 0.005 | 0.004 ± 0.006 | same |
| live-like crafts_distinct | 23.543 ± 0.623 | 25.427 ± 0.196 | better |
| live-like repeat_crafts | 0.728 ± 0.015 | 0.765 ± 0.009 | worse |
| live-like fail_pct | 19.775 ± 2.150 | 17.325 ± 1.430 | same |
| live-like tools_made | 0.800 ± 0.061 | 0.940 ± 0.110 | better |
| live-like rooms | 0.145 ± 0.160 | 0.258 ± 0.149 | same |
| live-like build_grade | 0.208 ± 0.170 | 0.388 ± 0.236 | same |
| live-like builds_5 | 0.000 ± 0.000 | 0.000 ± 0.000 | same |
| live-like goals_done | 0.586 ± 0.059 | 0.613 ± 0.061 | same |
| live-like goal_gather | 0.613 ± 0.203 | 0.802 ± 0.098 | same |
| live-like goal_craft | 0.789 ± 0.054 | 0.819 ± 0.216 | same |
| live-like goal_give | 0.951 ± 0.052 | 0.972 ± 0.088 | same |
| live-like goal_ask | 0.825 ± 0.218 | 0.844 ± 0.376 | same |
| live-like goal_stay | 0.734 ± 0.058 | 0.736 ± 0.060 | same |
| live-like goal_help | 0.384 ± 0.111 | 0.397 ± 0.032 | same |
| live-like goal_avoid | 0.306 ± 0.186 | 0.354 ± 0.066 | same |

verdict: keep the shipped body (targets: score; 21 min)
- **Result (r50):** best 7.03; gate against r47 (today's game): rooms 13% -> 27% (lush, better), build grade 0.22 ->
  0.43, tools better on live-like, score same (6.56 -> 6.71), repeat crafts worse (0.73 -> 0.77), no guard worse.
  **Shipped** for its builds (the owner's call: the smartest body, builds first).
- **Result (r51):** void as a build test. Its best was its first check (7.33 at +75k), then flat 6 checks. The
  policy never picked `build`: legal in 3515 decisions, picked 0, p = 0.0001 (a new action's row starts low, and at
  this entropy PPO never tries it). And its workers ran the venv's own older game (rl/vec.py, fixed in 91ae423).

## r52

- From: r50 best (shipped), `--init`, `--copy-rows build=pursue` (the new action starts as the macro it already uses).
- The workers run the pinned game (today's: roofs, ores and tool tiers; 200 actions, 13 more than r50 knew).
- Target: builds_5 up from 0; score and deaths no worse.
- **Result (r52):** build picked 2% of the times it is legal (r51: 0); a house of 5 checks in 4% of episodes at the
  first check, then 0; rooms 29% -> 10% by +1.6M. Stopped there: its last checkpoint is on today's game and uses build.

## r53

- From: r52 last, `--init`. Home-kit starts (--home-pct 20: a hut's pieces in the pack, rl/starts.py), so the build
  reward comes early enough to learn from. Target: builds_5 up from 0.
- **Result (r53):** stopped at +480k: with the kit in its pack, build picked in 0.4% of the decisions where it was
  legal, no hut; the build reward only moved when a ring closed.

## r54

- From: r53 last, `--init`, home-kit starts 20%. Changed: each piece of the blueprint laid pays 0.05 (judge.laid).
