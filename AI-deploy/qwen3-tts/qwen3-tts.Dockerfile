# 复用服务器已有的运行时镜像；需要编译 CUDA 扩展时可覆盖基础镜像。
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG APT_MIRROR=https://mirrors.ustc.edu.cn/ubuntu
ARG APT_SECURITY_MIRROR=https://mirrors.ustc.edu.cn/ubuntu
ARG APT_FALLBACK_MIRRORS="https://mirrors.tuna.tsinghua.edu.cn/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu https://archive.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu"
ARG PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
ARG PIP_FALLBACK_INDEX_URLS="https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.org/simple"

WORKDIR /app
COPY scripts/configure_apt.py scripts/preserve_torch.py ./scripts/
# 每轮从原始配置换源，保留发行版、组件及签名；第三方仓库保持不变。
# 备用项格式为普通仓库|安全仓库；更新或安装失败均尝试下一项。
RUN cp -a /etc/apt /tmp/apt-original && \
    ( installed=0; \
      for mirrors in "${APT_MIRROR}|${APT_SECURITY_MIRROR}" ${APT_FALLBACK_MIRRORS}; do \
        cp -a /tmp/apt-original/. /etc/apt/ && \
        rm -rf /var/lib/apt/lists/* || exit 1; \
        echo "尝试 apt 仓库：${mirrors}"; \
        APT_MIRROR="${mirrors%%|*}" APT_SECURITY_MIRROR="${mirrors#*|}" \
          python scripts/configure_apt.py || exit 1; \
        if apt-get -o APT::Update::Error-Mode=any update && \
           apt-get install -y --no-install-recommends sox libsndfile1 ffmpeg curl ca-certificates; then \
          installed=1; break; \
        fi; \
        echo "本轮 apt 失败；如有下一源则继续尝试，具体原因见上方日志。" >&2; \
      done; \
      test "$installed" = 1 ) && \
    rm -rf /var/lib/apt/lists/* /tmp/apt-original

COPY requirements.txt ./
# 从基础镜像生成约束，依赖冲突时失败，而不是重新安装另一套 PyTorch/CUDA。
# 任意安装失败均尝试下一源；全部失败时退出，不绕过后续校验。
RUN python scripts/preserve_torch.py /tmp/torch-constraints.txt && \
    ( installed=0; \
      for index in "${PIP_INDEX_URL}" ${PIP_FALLBACK_INDEX_URLS}; do \
        echo "尝试安装依赖：${index}"; \
        if python -m pip install --no-cache-dir --index-url "${index}" \
            --timeout 120 --retries 5 -c /tmp/torch-constraints.txt -r requirements.txt; then \
          installed=1; break; \
        fi; \
        echo "本轮安装失败；如有下一源则继续尝试，具体原因见 pip 日志。" >&2; \
      done; \
      test "$installed" = 1 ) && \
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
