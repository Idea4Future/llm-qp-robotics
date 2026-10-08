#!/usr/bin/env bash
# Foreground, loopback-only local inference server. Ctrl+C stops it.
set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OPTI_LLM_BACKEND="${OPTI_LLM_BACKEND:-cuda}"
OPTI_LLM_PORT="${OPTI_LLM_PORT:-8085}"
OPTI_LLM_CONTEXT="${OPTI_LLM_CONTEXT:-8192}"
if [[ "$OPTI_LLM_BACKEND" != cuda && "$OPTI_LLM_BACKEND" != cpu ]]; then
  echo 'OPTI_LLM_BACKEND must be cuda or cpu.' >&2
  exit 1
fi
if [[ ! "$OPTI_LLM_PORT" =~ ^[0-9]+$ ]] || (( OPTI_LLM_PORT < 1 || OPTI_LLM_PORT > 65535 )); then
  echo 'OPTI_LLM_PORT must be an integer in 1..65535.' >&2
  exit 1
fi
if [[ ! "$OPTI_LLM_CONTEXT" =~ ^[0-9]+$ ]] || (( OPTI_LLM_CONTEXT < 1024 )); then
  echo 'OPTI_LLM_CONTEXT must be an integer of at least 1024.' >&2
  exit 1
fi
OPTI_BINARY="$PROJECT_ROOT/third_party/llm_runtime/build_$OPTI_LLM_BACKEND/bin/llama-server"
OPTI_MODEL="$PROJECT_ROOT/third_party/llm_models/qwen3_4b_q4_k_m/Qwen3-4B-Q4_K_M.gguf"
if [[ ! -x "$OPTI_BINARY" || ! -f "$OPTI_MODEL" ]]; then
  echo 'Pinned local runtime/model are not ready. Run scripts/local_llm_setup.py first.' >&2
  exit 1
fi
mkdir -p "$PROJECT_ROOT/logs/local_llm"
OPTI_LOG="$PROJECT_ROOT/logs/local_llm/server_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S).log"
OPTI_GPU_LAYERS=all
if [[ "$OPTI_LLM_BACKEND" == cpu ]]; then OPTI_GPU_LAYERS=0; fi
echo "Local LLM: http://127.0.0.1:$OPTI_LLM_PORT/v1"
echo "Log: $OPTI_LOG"
echo 'Server runs in the foreground; Ctrl+C stops it.'
exec "$OPTI_BINARY" --model "$OPTI_MODEL" --host 127.0.0.1 --port "$OPTI_LLM_PORT" \
  --ctx-size "$OPTI_LLM_CONTEXT" --parallel 1 --threads 6 --jinja --reasoning off \
  --n-gpu-layers "$OPTI_GPU_LAYERS" > "$OPTI_LOG" 2>&1
