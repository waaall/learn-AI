"""在启动 GPU 服务前准备模型：本地优先、缓存复用、有限重试和官方源回退。"""
import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from download_worker import DOWNLOAD_FAILURE, OFFICIAL_ENDPOINT
from model_files import validate_model_directory
from settings import Settings
from startup_logging import configure_logging, startup_stage

logger = logging.getLogger("qwen3_tts.prepare")


def enabled(name, default="0"):
    value = os.getenv(name, default).strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError(f"{name} 必须是布尔值")
    return value in {"1", "true", "yes", "on"}


def positive_int(name, default):
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value


def endpoint_sources():
    primary = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    parsed = urlsplit(primary)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("HF_ENDPOINT 必须是无凭据、无查询参数的 HTTPS 地址")
    sources = [primary]
    if enabled("HF_FALLBACK_TO_OFFICIAL", "1") and primary != OFFICIAL_ENDPOINT:
        sources.append(OFFICIAL_ENDPOINT)
    return sources


def cached_snapshot(repo_id, revision, cache_dir):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return Path(snapshot_download(repo_id, revision=revision, cache_dir=cache_dir,
                                      local_files_only=True, token=False))
    except LocalEntryNotFoundError as exc:
        # 新版 Hub 对不完整 snapshot 抛错，但仍可能提供部分缓存所在目录。
        partial = getattr(exc, "snapshot_path", None)
        return Path(partial) if partial else None


def cached_model(repo_id, revision, cache_dir):
    path = cached_snapshot(repo_id, revision, cache_dir)
    if path is None:
        logger.info("尚未找到该版本的本地 snapshot")
        return None
    try:
        return validate_model_directory(str(path))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        logger.info("缓存尚不可直接使用：%s", type(exc).__name__)
        return None


def prepare():
    local_path = os.getenv("MODEL_LOCAL_PATH", "").strip()
    if local_path:
        logger.info("使用外部模型目录；不会访问镜像或官方源")
        return validate_model_directory(local_path)

    repo_id = Settings.from_env().model_id
    revision = os.getenv("MODEL_REVISION", "main")
    cache_dir = os.getenv("HF_HUB_CACHE") or str(
        Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub")
    # 保存已验证的 snapshot 路径，不依赖下载提交哈希时是否更新了 Hub 的分支引用。
    selection_key = hashlib.sha256(f"{repo_id}\n{revision}".encode()).hexdigest()
    selection_file = Path(cache_dir) / ".qwen-prepared" / selection_key
    if selection_file.is_file():
        try:
            selected = validate_model_directory(selection_file.read_text().strip())
        except (ValueError, KeyError, FileNotFoundError):
            logger.info("之前准备的 snapshot 已不完整，将重新检查缓存")
        else:
            logger.info("复用之前准备好的模型：%s", selected)
            return selected
    cached = cached_model(repo_id, revision, cache_dir)
    if cached:
        logger.info("完整运行文件已在缓存，直接使用：%s", cached)
        return cached
    if enabled("HF_HUB_OFFLINE"):
        raise RuntimeError("离线模式下缓存不完整，请补齐文件或配置 MODEL_LOCAL_PATH")

    attempts = positive_int("MODEL_DOWNLOAD_ATTEMPTS", 2)
    timeout = positive_int("MODEL_DOWNLOAD_TIMEOUT", 900)
    http_timeout = positive_int("HF_HUB_DOWNLOAD_TIMEOUT", 30)
    workers = positive_int("MODEL_DOWNLOAD_WORKERS", 4)
    token_file = os.getenv("HF_TOKEN_FILE", "").strip()
    if token_file and not Path(token_file).is_file():
        raise ValueError("HF_TOKEN_FILE 不存在；请检查只读凭据挂载")

    with tempfile.TemporaryDirectory(prefix="qwen-download-") as temporary:
        output = Path(temporary) / "model-path"
        revision_file = Path(temporary) / "revision"
        # 如果已有部分 snapshot，沿用其提交，避免 main 更新后重新下载另一版本。
        partial = cached_snapshot(repo_id, revision, cache_dir)
        if partial and re.fullmatch(r"[0-9a-fA-F]{40}", partial.name):
            revision_file.write_text(partial.name)

        for endpoint in endpoint_sources():
            for attempt in range(1, attempts + 1):
                logger.info("源=%s；尝试 %d/%d；本次最长 %d 秒", endpoint, attempt, attempts, timeout)
                config = dict(repo_id=repo_id, revision=revision, cache_dir=cache_dir,
                              endpoint=endpoint, workers=workers, http_timeout=http_timeout,
                              token_file=token_file if endpoint == OFFICIAL_ENDPOINT else "",
                              output=str(output), revision_file=str(revision_file))
                env = os.environ.copy()
                # 禁止自动读取环境变量 Token；仅 worker 在官方源显式读取挂载文件。
                for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                    env.pop(key, None)
                env.update(HF_ENDPOINT=endpoint, HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                           HF_HUB_DISABLE_XET=os.getenv("HF_HUB_DISABLE_XET", "1"),
                           HF_HUB_DOWNLOAD_TIMEOUT=str(http_timeout),
                           HF_HUB_ETAG_TIMEOUT=str(http_timeout))
                try:
                    result = subprocess.run(
                        [sys.executable, str(Path(__file__).with_name("download_worker.py"))],
                        input=json.dumps(config), text=True, env=env, timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    logger.warning("当前下载进程超时，已终止；保留已写入缓存的文件")
                else:
                    if result.returncode == 0:
                        prepared = validate_model_directory(output.read_text())
                        selection_file.parent.mkdir(parents=True, exist_ok=True)
                        temporary_selection = selection_file.with_suffix(".tmp")
                        temporary_selection.write_text(str(prepared))
                        temporary_selection.replace(selection_file)
                        return prepared
                    if result.returncode != DOWNLOAD_FAILURE:
                        raise RuntimeError("模型准备发生非下载错误，不进行换源；见上一条日志")
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 10))
            logger.warning("当前源尝试已用尽，将尝试下一配置源（若存在）")
    raise RuntimeError("所有下载源均失败；可检查网络、配置官方 Token，或挂载完整本地模型。缓存已保留。")


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with startup_stage("准备模型文件（本地目录 / 缓存 / 镜像 / 官方回退）"):
        args.output.write_text(str(prepare()))
