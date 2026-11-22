#!/bin/sh
# AI players on the local model server: tools/ai_players.sh [--n 8] [--model 30b] [--pick gatsu0,senku0] ...
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
PYTHONPATH=src exec "$PY" -m megastructure.players "$@"
