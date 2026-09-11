FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git sox libsox-dev ffmpeg curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    faster-qwen3-tts \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    soundfile

COPY api/ ./api/
COPY docker/scripts/entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    DTYPE=bfloat16 \
    DEVICE=cuda:0 \
    ATTN_IMPLEMENTATION=sdpa \
    PORT=8080 \
    HF_HOME=/root/.cache/huggingface \
    HF_ENDPOINT=https://huggingface.co \
    PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
