#!/bin/bash
# 单进程提供预设音色服务，避免多个工作进程重复占用显存。
set -euo pipefail
: "${PORT:=8080}"
echo "=== Qwen3-TTS: ${MODEL_ID:-default} / ${DTYPE:-bfloat16} / ${ATTN_IMPLEMENTATION:-sdpa} ==="
exec python -m uvicorn main:app \
    --host 0.0.0.0 --port "${PORT}" --workers 1 \
    --log-level info --timeout-keep-alive 65
