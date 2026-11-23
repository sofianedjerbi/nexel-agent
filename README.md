# Nexel Agent

AI players for [Nexel Worlds](https://github.com/sofianedjerbi/nexel-worlds): programs that join the game through its
agent API (JSON over a websocket, the same rules as a person) and play it.

- **Thinkers** (`thinker.py`, `planner.py`, `react.py`, `fastprompt.py`, `plans.py`, ...): players driven by a language
  model running locally (MLX on Apple Silicon by default; any OpenAI-compatible chat endpoint works), with memory
  grounded in what actually happened, goals, a watchdog for loops, and a scripted body that keeps acting while the
  model thinks.
- **Personas** (`personas.json`) and the launcher `players.py`: one command for a shared local model server plus N players.
- **Scripted bots** (`bots.py`, `policy.py`): cheap players for load and smoke tests.
- **Skill forge** (`skills/`): the model writes small skills, checked in a sandboxed copy of the game before they are kept.
- **Reinforcement learning** (`rl/`): an env over the game's own simulation (Crafter-style achievements), an action
  mask, a numpy policy and a live RL player. Work in progress: imported as committed on 2026-11-23 while a training
  run was still going (the trainer side runs in its own Python 3.12 env with torch, stable-baselines3, sb3-contrib
  and gymnasium: `rl/vec.py`, `rl/train.py`).

The game itself (server, simulation, client, art install) is in nexel-worlds; this package imports it as
`megastructure` (the game's first name) and never the other way round.

## Install

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

## Run

Start a game first (in nexel-worlds: `python -m megastructure.server`, http://127.0.0.1:7777/), then:

```sh
python -m nexel_agent.players                       # 2 players on Qwen3-8B, the game at :7777
python -m nexel_agent.players --n 8 --model 30b --minutes 60      # models: 4b 8b 14b 30b
python -m nexel_agent.players --pick gatsu0,senku0  # --list shows the roster (src/nexel_agent/personas.json)
python -m nexel_agent.bots --n 8 --minutes 30       # scripted bots, no model
```

`players.py` starts the shared model server on :8080 (or reuses one answering there), then the players a few seconds
apart; logs go to `agents/<name>.log`. Ctrl-C stops them all and the model server it started. RAM: about 3 GB for 4b,
5 GB for 8b, 9 GB for 14b, 17 GB for 30b, plus ~100 MB a player; one model serves the players in turn, so keep N
small (2 to 4) on a laptop. A world that asks who players are lets programs on its machine in with its key:
`MEGASTRUCTURE_AGENT_KEY=$(cat <game>/saves/default/agent.key)`.

Tools: `tools/ai_players.sh` (= `players`), `tools/llm_bench.py` (model call timings on real observations),
`tools/soak_report.py` (a run's decisions.jsonl), `tools/qa_visitor.py`, `tools/rl_soak_visitor.py` and `tools/chat_probe.py` (scripted visitors
for a QA copy of a world), `tools/forge_night.sh` (the skill forge, at night, on a running model server).

## Tests

```sh
pip install pytest pytest-asyncio pytest-xdist
pytest -q          # runs the game's simulation in-process: no server, no model, no art needed
```

## License

MIT (`LICENSE`), code only. The game's Sprout Lands art is not part of either repository (nexel-worlds README, "Art").
