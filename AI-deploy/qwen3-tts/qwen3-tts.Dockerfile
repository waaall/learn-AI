# 复用服务器已有的运行时镜像；需要编译 CUDA 扩展时可覆盖基础镜像。
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG APT_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app
COPY scripts/configure_apt.py scripts/preserve_torch.py ./scripts/
# 仅替换 Ubuntu 仓库地址，保留发行版、组件及签名设置。
RUN python scripts/configure_apt.py && \
    apt-get update && \
    apt-get install -y --no-install-recommends sox libsndfile1 ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# 从基础镜像生成约束，依赖冲突时失败，而不是重新安装另一套 PyTorch/CUDA。
RUN python scripts/preserve_torch.py /tmp/torch-constraints.txt && \
    python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
        --timeout 120 --retries 5 -c /tmp/torch-constraints.txt -r requirements.txt && \
    python scripts/preserve_torch.py /tmp/torch-constraints.txt --verify && \
    python -m pip check && \
    python -c "from faster_qwen3_tts import FasterQwen3TTS"

COPY main.py settings.py audio_utils.py tts_service.py startup_logging.py entrypoint.sh ./
COPY prepare_model.py download_worker.py model_files.py ./
RUN chmod +x entrypoint.sh

ENV MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    DTYPE=bfloat16 DEVICE=cuda:0 ATTN_IMPLEMENTATION=sdpa \
    PORT=8080 HF_HOME=/root/.cache/huggingface \
    HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1

EXPOSE 8080
# 首次下载模型可能较慢；健康检查不负责自动重启容器。
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1
ENTRYPOINT ["./entrypoint.sh"]
