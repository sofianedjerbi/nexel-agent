#!/bin/zsh
# The skill forge, at night, on the model server already running for the players (MODEL_URL, players.py's :8080):
# the forge writes again what play found at fault (agents/*/decisions.jsonl) and any curriculum skill the library
# lacks. It never starts or stops a model server. A memory guard stops the forge (not the server).
#
#   tools/forge_night.sh                 # from the repo root; cron: 0 3 * * * cd ~/nexel-agent && tools/forge_night.sh
#
# Guard (owner's rule): stop if free memory < 12%, swap > 6 GB, or swap grows > 500 MB over the run.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
MODEL_URL=${MODEL_URL:-http://127.0.0.1:8080/v1/chat/completions}
DATA=${DATA:-agents}
LIB=${LIB:-$DATA/_skills}
LOG=${LOG:-$DATA/forge_night.log}
MINFREE=${MINFREE:-12}; MAXSWAP=${MAXSWAP:-6144}; GROW=${GROW:-500}

swap() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}' }
free() { memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%' }
if ! curl -s -m 5 "${MODEL_URL%/chat/completions}/models" >/dev/null; then
  echo "$(date) no model server answers at $MODEL_URL: no forge tonight" >> $LOG; exit 0
fi
S0=$(swap)
echo "$(date) forge on $MODEL_URL, free $(free)%, swap ${S0}M" >> $LOG
PYTHONPATH=src $PY -m nexel_agent.skills.forge --model-url $MODEL_URL --lib $LIB --from-play $DATA --tries 6 >> $LOG 2>&1 &
FP=$!
trap 'kill $FP 2>/dev/null' EXIT
while kill -0 $FP 2>/dev/null; do
  s=$(swap); f=$(free)
  if [ "$s" -gt "$MAXSWAP" ] || [ "$f" -lt "$MINFREE" ] || [ $((s - S0)) -gt "$GROW" ]; then
    echo "$(date) guard: free ${f}%, swap ${s}M (start ${S0}M): forge stopped" >> $LOG
    kill $FP; break
  fi
  sleep 10
done
wait $FP 2>/dev/null
echo "$(date) forge done" >> $LOG
