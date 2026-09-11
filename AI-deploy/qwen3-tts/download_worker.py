"""在独立进程中下载单个源，隔离 Hub 环境变量，并限制凭据使用范围。"""
import json
import logging
import re
import sys
from pathlib import Path

from model_files import validate_model_directory
from startup_logging import configure_logging

OFFICIAL_ENDPOINT = "https://huggingface.co"
DOWNLOAD_FAILURE = 21
CONFIGURATION_FAILURE = 22
logger = logging.getLogger("qwen3_tts.download")


def download(config):
    # 环境变量由父进程在创建本进程前设置，避免修改已导入库的常量。
    from huggingface_hub import HfApi, snapshot_download

    endpoint = config["endpoint"]
    token = False
    token_file = config.get("token_file")
    if endpoint == OFFICIAL_ENDPOINT and token_file:
        token = Path(token_file).read_text().strip()
        if not token:
            raise ValueError("配置的 Token 文件为空")
    logger.info("下载源=%s；认证=%s", endpoint, "已提供官方 Token" if token else "匿名")

    revision_file = Path(config["revision_file"])
    if revision_file.exists():
        revision = revision_file.read_text().strip()
    elif re.fullmatch(r"[0-9a-fA-F]{40}", config["revision"]):
        revision = config["revision"]
    else:
        # 首次解析得到提交哈希后保留到本轮结束；回退不切换模型版本。
        revision = HfApi(endpoint=endpoint, token=token).model_info(
            config["repo_id"], revision=config["revision"], timeout=config["http_timeout"],
        ).sha
    if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("未获取到有效的模型提交哈希")
    revision_file.write_text(revision)
    logger.info("准备下载 revision=%s；已缓存文件会复用", revision)
    path = snapshot_download(
        repo_id=config["repo_id"], revision=revision, endpoint=endpoint,
        token=token, cache_dir=config["cache_dir"],
        max_workers=config["workers"], etag_timeout=config["http_timeout"],
    )
    path = validate_model_directory(path)
    Path(config["output"]).write_text(str(path))


def is_download_failure(exc):
    import httpx
    from huggingface_hub import errors

    error_types = [httpx.HTTPError, errors.HfHubHTTPError, errors.LocalEntryNotFoundError,
                   errors.FileMetadataError]
    xet_error = getattr(errors, "XetDownloadError", None)
    if xet_error is not None:
        error_types.append(xet_error)
    if isinstance(exc, tuple(error_types)):
        return True
    # Xet 的错误未统一包装为 HTTP 异常，只识别其下载错误，不吞掉任意 RuntimeError。
    return isinstance(exc, RuntimeError) and any(
        marker in str(exc).lower() for marker in ("cas client error", "file reconstruction", "xet")
    )


if __name__ == "__main__":
    configure_logging()
    # 禁止 HTTP DEBUG 日志输出认证头或签名 URL。
    for name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        download(json.load(sys.stdin))
    except Exception as exc:
        retryable = is_download_failure(exc)
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # 下载异常原文可能含签名地址；只记录类型和状态码，不打印原始响应。
        if retryable:
            logger.error("下载失败：类型=%s，HTTP=%s；将由父进程决定重试或回退",
                         type(exc).__name__, status or "不可用")
        else:
            logger.error("本地配置或文件校验失败：%s；%s", type(exc).__name__, exc)
        sys.exit(DOWNLOAD_FAILURE if retryable else CONFIGURATION_FAILURE)
