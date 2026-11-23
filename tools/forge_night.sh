#!/bin/zsh
# The skill forge, at night: one model server (the 30B, nothing else loaded), the forge on what play found at fault
# (agents/*/decisions.jsonl) plus any curriculum skill the library lacks, a memory guard, and the server stopped after.
#
#   tools/forge_night.sh                 # from the repo root; cron: 0 3 * * * cd ~/megastructure && tools/forge_night.sh
#
# Guard (owner's rule): stop if free memory < 12%, swap > 6 GB, or swap grows > 500 MB over the run. It waits while
# another model process runs (one model at a time) and while the 1-minute load is 6 or more.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit}
PORT=${PORT:-8091}
DATA=${DATA:-agents}
LIB=${LIB:-$DATA/_skills}
LOG=${LOG:-$DATA/forge_night.log}
MINFREE=${MINFREE:-12}; MAXSWAP=${MAXSWAP:-6144}; GROW=${GROW:-500}

swap() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}' }
free() { memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%' }
for i in $(seq 1 60); do                         # (at most an hour of waiting)
  if [ -z "$(pgrep -f '[Pp]ython.*(mlx_lm|proto/skillcode)')" ] && [ "$(sysctl -n vm.loadavg | awk '{print int($2)}')" -lt 6 ]; then
    break
  fi
  sleep 60
done
[ -n "$(pgrep -f '[Pp]ython.*(mlx_lm|proto/skillcode)')" ] && { echo "$(date) another model is loaded: no forge tonight" >> $LOG; exit 0; }

S0=$(swap)
echo "$(date) forge: $MODEL, free $(free)%, swap ${S0}M" >> $LOG
$PY -m mlx_lm server --model $MODEL --host 127.0.0.1 --port $PORT --chat-template-args '{"enable_thinking":false}' \
    --prompt-cache-bytes 300000000 >> $LOG 2>&1 &
MP=$!; FP=""
trap 'kill $MP $FP 2>/dev/null' EXIT
until curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null; do sleep 2; kill -0 $MP || exit 1; done
PYTHONPATH=src $PY -m megastructure.skills.forge --model-url http://127.0.0.1:$PORT/v1/chat/completions --lib $LIB \
    --from-play $DATA --tries 6 >> $LOG 2>&1 &
FP=$!
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
