# Contributing

## Commit messages: Conventional Commits

Every commit on main follows [Conventional Commits](https://www.conventionalcommits.org): CI checks each one
(`commitlint.config.mjs`: the conventional rules; body lines may be long).

    <type>(<scope>): <subject in lower case, no final full stop, about 72 characters at most>

    <body: what changed and why, as long as it needs>

- **type**: `feat` (new behaviour), `fix` (a bug), `perf`, `refactor` (no behaviour change), `test`, `docs`,
  `build` (pyproject, uv.lock), `ci` (.github), `chore` (tools, anything else), `style`, `revert`.
- **scope**: the main part touched: `rl` (the body, its env and training), `mind` (the planner), `voice` (its
  words), `facts`, `players`, `brain`, `tests`, `tools`, `deps`.
- Breaking changes (`feat!:` or a `BREAKING CHANGE:` footer): a minor release while the version is 0.x.

Optional hooks (ruff, and the message check at commit time): `uvx pre-commit install`.

## Releases: tag to release

Nobody bumps a version by hand.

1. A push to main runs `.github/workflows/semantic-release.yml`. python-semantic-release reads the commits
   since the last tag `vX.Y.Z`: a `feat` makes the next minor, a `fix` or `perf` the next patch (while the
   version is 0.x; `major_on_zero = false`). Other types make no release.
2. When there is one, it sets `project.version` in pyproject.toml (and uv.lock), adds it to CHANGELOG.md,
   commits `chore(release): vX.Y.Z`, tags `vX.Y.Z`, pushes, and makes the GitHub Release with the notes.
3. `.github/workflows/release.yml` builds the sdist and wheel (`uv build`) for the tag and attaches them to the
   release. Nothing goes to PyPI.

Preview what the next release would be: `uvx --from python-semantic-release semantic-release --noop version --print`.

## Checks

`.github/workflows/ci.yml` (pull requests and main): ruff, the unit tests (`uv run pytest tests -q`,
Python 3.14), commit messages. `security.yml` (pull
requests, main, weekly): CodeQL, uv audit, gitleaks, bandit. Not blocking for now: ruff format,
the ruff rules listed in `ignore` in pyproject.toml, and bandit.

## Working on the code

A program that joins [Nexel Worlds](https://github.com/sofianedjerbi/nexel-worlds) through its agent API (JSON over a
websocket, the same rules as a person) and plays it. The game itself (server, simulation, client, art install) is in
nexel-worlds; this package imports it as `megastructure` and never the other way round.

### What is where

- **The player** (`rl/live.py`): a body trained by reinforcement learning (`models/rook.npz`, numpy only) chooses
  every action. Over it, a local language model (MLX on Apple Silicon by default; any OpenAI-compatible chat endpoint
  works) plans (`rl/mind.py`: the goal the body is after, its aims, who is an ally) and talks (`rl/voice.py`). With
  no model server it plays on in silence.
- **Facts** (`rl/facts.py`): the one place an observation is read: a parser per game event, one distance.
- **Training** (`rl/core.py`, `rl/actions.py`, `rl/encode.py`, `rl/achieve.py`, ...): an env over the game's own
  simulation (Crafter-style achievements), an action mask, the numpy policy. The trainer side runs in its own Python
  3.12 env with torch, stable-baselines3, sb3-contrib and gymnasium (`rl/vec.py`, `rl/train.py`).
- **Personas** (`personas.json`) and the launcher `players.py`: one command for a shared local model server plus N
  players.

### Install

Python 3.14.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[mlx]"            # pulls nexel-worlds from GitHub; [mlx] = the local model server (Apple Silicon)
```

Working on both repositories side by side (`../nexel-worlds` checked out next to this one):

```sh
pip install -e ../nexel-worlds
pip install --no-deps -e . && pip install httpx numpy websockets "mlx-lm; sys_platform == 'darwin'"
# or with uv: uncomment the `nexel-worlds = { path = ... }` line under [tool.uv.sources] in pyproject.toml, then `uv sync`
```

### Run

Start a game first (in nexel-worlds: `nexel-worlds`, http://127.0.0.1:7777/), then:

```sh
nexel-agent                                 # Rook: the shipped body (models/rook.npz); plans and words from a model on :8080
nexel-agent --name Ash --persona "..."      # --policy <weights.npz> for another body, --model-url "" to play in silence
nexel-agent-players --n 2                   # the model server on :8080 and 2 players of the roster (personas.json)
nexel-agent-players --pick gatsu0,senku0 --model 30b --minutes 60      # --list shows the roster; models: 4b 8b 14b 30b
```

(`uv run nexel-agent` from a checkout; `python -m nexel_agent` is the same command.) The body's weights are read from
the checkout's `models/`: install with `pip install -e .` or run with uv, or pass `--policy`.

`nexel-agent-players` starts the shared model server on :8080 (or reuses one answering there), then the players a few
seconds apart; logs go to `agents/<name>.log` and `agents/<name>.rl.jsonl`. Ctrl-C stops them all and the model server
it started. RAM: about 3 GB for 4b, 5 GB for 8b, 9 GB for 14b, 17 GB for 30b, plus ~100 MB a player; one model serves
the players in turn, so keep N small (2 to 4) on a laptop. Serve Qwen3-30B-A3B-Instruct-2507 (4-bit, `--model 30b`):
measured with training running, it replies fastest (p90 0.6 s) and stays on topic; on a Mac with less than ~20 GB free
use `4b` (`rl/voice.py` MODEL, FALLBACK_MODEL). A world that asks who players are lets programs on its machine in with
its key: `MEGASTRUCTURE_AGENT_KEY=$(cat <game>/saves/default/agent.key)`.

Tools: `tools/ai_players.sh` (= `nexel-agent-players`), `tools/rl_soak_visitor.py`, `tools/rl_social_soak.py` and
`tools/rl_talk_soak.py` (scripted visitors for a QA copy of a world: they hit, ask, talk, and tell what the player did).

### The brain view

`tools/brain/index.html` shows what an RL player sees, thinks, decides and remembers, live from
`nexel-agent --brain 8790` (open it with `?feed=http://127.0.0.1:8790/feed`) or from a
recording (`?replay=demo_replay.jsonl`, made by `tools/brain/from_demo.py` from a run's logs). The feed:
`docs/brain_feed.md`.

### Tests

```sh
pip install pytest pytest-asyncio pytest-xdist
pytest -q          # runs the game's simulation in-process: no server, no model, no art needed
```
