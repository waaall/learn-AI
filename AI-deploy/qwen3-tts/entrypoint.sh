#!/bin/bash
# 单进程提供预设音色服务，避免多个工作进程重复占用显存。
set -euo pipefail
: "${PORT:=8080}"
echo "=== Qwen3-TTS: ${MODEL_ID:-default} / ${DTYPE:-bfloat16} / ${ATTN_IMPLEMENTATION:-sdpa} ==="
# 下载子进程先完成文件准备；随后在全新进程中导入离线模型库。
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
prepared_path=$(mktemp)
trap 'rm -f "$prepared_path"' EXIT
python prepare_model.py --output "$prepared_path"
export MODEL_LOAD_PATH="$(cat "$prepared_path")"
rm -f "$prepared_path"
trap - EXIT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec python -m uvicorn main:app \
    --host 0.0.0.0 --port "${PORT}" --workers 1 \
    --log-level info --timeout-keep-alive 65
