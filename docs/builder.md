# The builder: how the AI player builds a home, and how it is judged

For the next agent who works on building (#25 and its parts). It says what each piece does and where it lives;
the code's own docstrings say the rest. As of r61 (2026-11-26).

## The parts at a glance

| part | what it does | where |
|---|---|---|
| judge v1 `verdict` | does the build work? 6 checks | `src/nexel_agent/rl/judge.py` (#26) |
| judge v2 `sense` | does it make sense? 8 checks | same file (#31) |
| judge v3 `looks` | does it look good? 8 builder's rules | same file (#32, part 1) |
| the `build` verb | a hut's blueprint, laid step by step | `src/nexel_agent/rl/build.py` (#28) |
| the build reward | `judge.Potential`: grade and pieces, potential-based | `rl/judge.py` (#29) |
| home-kit starts | episodes that start with a hut's pieces in the pack | `rl/starts.py`, `--home-pct` (train.py) |
| KPIs | `build_grade`, `builds_5` in the sim; the live builds table | `tools/kpis.py`, `docs/kpis.md` |
| live scorecard | a run of the player on a QA copy, its rooms judged | `tools/rl_live_scorecard.py`, `tools/judge_builds.py` |
| review page | every room drawn as tiles (and renders) beside its reports | `tools/build_review.py` (#33) |
| renders, VLM rubric | judge v3 part 2: renders and a vision model's rubric | `tools/render_build.py`, `tools/build_vlm.py` (#32) |

## The judge: three versions, one report

`judge.report(world, x, y)` reads the build round a tile from the world as facts: is it closed (a flood fill from the
tile that crosses no piece the game counts as enclosing, `Simulation.ENCLOSE`, up to `CAP` tiles), its area, box,
doors, walls by material, the share of the inside under a roof, the lights within reach, what stands inside. The
game decides what closes a room and what gives light; the judge only reads it. Each version adds checks on that
report. A grade is the number of checks passed: no weights.

- **v1, `verdict`: does it work?** `CHECKS = closed, door, roofed (90 % of the inside), lit, bed, room_enough (4
  tiles or more)`. `told()` puts it in words for the mind's facts ("closed, a door, no roof (40%) ... grade 2 of 6").
- **v2, `sense`: does it make sense?** `SENSE = chest, station (a workbench or a campfire), seat, reachable (every
  core thing reached from a door), door_out (a door onto walkable ground), fit (6 to 40 tiles), tidy (no land raised
  inside), light_by_bed`.
- **v3, `looks`: does it look good?** `LOOKS = material (one wall material, or a deliberate second), rectangles,
  straight (no single-tile jogs), roof (with an overhang), symmetric (about the door), windows (fences in the wall),
  path (cobble from the door), no_stray`. v3 reads fences as closing a room, though the game's shelter does not
  (asked of the game's owner: nexel-worlds#135).
- **v3 part 2, a vision model's view.** `tools/render_build.py` draws each room with the game's own renderer
  (three quarters, and from over it with the roof cut away), and `tools/build_vlm.py` puts the renders to a VLM with a
  fixed rubric (neat, cosy, purposeful, 1-5 with a reason). It checks v3's rules, never alone, and joins the reward
  only once it agrees with them on 30 calibration builds (Spearman 0.6 or more, #32).

## The `build` verb and its blueprint

`rl/build.py` is the "how" of a home; the policy chooses the "when" (build is one action among the others). The
blueprint is a hut (`blueprint()`, `PLAN`): a 3 x 3 inside walled all round, a door in the middle of the south side,
a campfire and a bed in two corners, a roof over all of it, laid walls first, then the door, the furniture, the roof.
The home is chosen once (`site`: the nearest spot in view where the whole hut stands on flat open ground, clear of a
hill's rim) and kept in `social.home`; it is given up only past `FAR` (64) tiles. `next_step` gives the next tool
call; one `build` pick walks to the middle and lays every piece the pack holds, in the plan's order, as one run that
is done to its end (r60, r61). A piece the pack lacks is made, or what it takes gathered, through `pursue`.

## The reward

`judge.Potential` pays as the build changes, potential-based (as `rl/progress.py`): the potential is
`UNIT (0.25) x the v1 grade + PIECE (0.05) x the blueprint's pieces laid (judge.laid)`, and each step pays its
change, so a wall torn down pays back what it earned. It is measured at one place, the home (else the last closed
build it stood in), so walking out of the room is no loss, and read only after a place or a dig. The per-piece pay
(r54) makes a ring of walls pay as it rises, not only once it closes. Only v1 is in the reward today; v2 and v3 are
reported, and v3 joins it once part 2 is calibrated.

## Home-kit starts

`starts.with_a_home_kit` starts an episode with a hut's pieces in the pack, as a player who came back with them:
the walls, the door, a bed, a fire, the roof; now and then short of the roof, the bed or the fire (never of a wall or
the door: r58 laid 28 pieces round a gap and left to make the door). `train.py --home-pct 20` gives 20 % of the
training starts a kit, so the build reward comes early enough to learn from (r53 on).

## KPIs

In the sim, from train.py's held-out checks (the `build=<grade>/<share>` of each `[check]` line):

- `build_grade`: the best build's v1 grade, mean per episode;
- `builds_5`: the share of episodes whose best build passes 5 of the 6 v1 checks or more: the builder's target;
- `rooms`: the share of episodes with a closed room.

Live, from a scorecard run (`tools/rl_live_scorecard.py`, a QA copy of a world with scripted visitors, kept in
`rl_runs/score/<run>/`): `builds_judged, best_grade, best_sense, best_looks`, its closed rooms judged by the three
versions at the end, beside the real players' rooms in the same copy, the reference. `python tools/kpis.py live
rl_runs/score/<run>` adds them to `docs/kpis.jsonl`; `docs/kpis.md` shows the table ("Live builds").

To look at the builds themselves: `python tools/build_review.py` writes `~/nexel/local/builds/index.html`, every
room of the saved runs drawn tile by tile beside its three reports, and its renders where `render_build.py` made them.

## The run history: r51 to r61

Each run started from the one before (`--init`); the full entries are in `rl_runs/LEDGER.md`. The target of all of
them: `builds_5` above 0. None got there yet; what stopped each one:

| run | changed | what blocked it |
|---|---|---|
| r51 | the `build` verb and the build reward (0.25 a check) arrive | build never picked (0 %) |
| r52 | the new action starts as `pursue` (`--copy-rows build=pursue`) | build picked 2 % of the times legal; a 5-check house in 4 % of episodes at the first check, then 0; rooms 29 % -> 10 % |
| r53 | home-kit starts (20 %) | build picked 0.4 % of the time; the reward moved only when a ring closed |
| r54 | each blueprint piece laid pays 0.05 | still 0.5 %, one piece a pick |
| r55 | one pick lays every piece the pack holds | picked 4-5 times an episode, but a pick only walked to the middle; the next came hundreds of decisions later |
| r56 | one pick walks to the middle and lays all it can | a kit short of a door sent it 30 tiles off and it gave the home up (FAR 30) |
| r57 | a home is given up only past 64 tiles | its own pits made the spot a hill's rim, and huts were refused (fixed: a spot dug into is left) |
| r58 | input 669 -> 688 (ores, ingots, tool tiers, #36) | a kit without a door: a ring with a gap, an errand, a new spot past FAR |
| r59 | a kit is short only of roof, bed or fire | the home moved each time it walked 9 tiles off (out-of-view tiles read as lower ground); build alone from a kit: 6/6 on 3 of 6 worlds, 5/6 on the others |
| r60 | a home is kept out of view's edge; a pit in a begun home is filled | a laying run cut after a few pieces by the next decision (the 5 s wait, a newcomer in view) |
| r61 | a laying run is one choice, done to its end | running at the time of writing |

The pattern: each run fixed what kept a hut from being finished once build was picked; the scripted verb alone
finishes it from a kit on every world tried (r59). What is left is the policy choosing build, and finishing.
