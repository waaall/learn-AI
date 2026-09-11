"""将启动阶段与等待状态输出到标准输出，供 Docker 日志直接查看。"""
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("qwen3_tts.startup")


def configure_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"不支持的 LOG_LEVEL: {level_name}")
    logging.basicConfig(
        level=level, stream=sys.stdout,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger().setLevel(level)
    # 模型包装库的阶段日志通过根日志器输出，不启用 HTTP 调试日志。
    logging.getLogger("faster_qwen3_tts").setLevel(level)


def cache_status():
    root = Path(os.getenv("HF_HUB_CACHE") or
                str(Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"))
    total, incomplete = 0, 0
    try:
        # 不重复统计 snapshot 软链接；这是整个缓存的观察值，不是下载完成百分比。
        for directory, _dirs, files in os.walk(root):
            for name in files:
                path = Path(directory) / name
                try:
                    if path.is_symlink():
                        continue
                    total += path.stat().st_size
                    incomplete += int(name.endswith(".incomplete"))
                except OSError:
                    continue
        return f"Hub 缓存合计 {total / 1024 ** 3:.2f} GiB，临时下载文件 {incomplete} 个"
    except OSError as exc:
        return f"缓存状态暂不可读：{type(exc).__name__}"


@contextmanager
def startup_stage(name):
    interval = float(os.getenv("STARTUP_LOG_INTERVAL", "30"))
    if interval <= 0:
        raise ValueError("STARTUP_LOG_INTERVAL 必须大于 0")
    started = time.monotonic()
    stop = threading.Event()
    logger.info("开始：%s", name)

    def heartbeat():
        while not stop.wait(interval):
            logger.info("仍在执行：%s；已等待 %.0f 秒；%s。状态日志不代表任务一定有进展。",
                        name, time.monotonic() - started, cache_status())

    worker = threading.Thread(target=heartbeat, name="startup-status", daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        logger.exception("失败：%s；耗时 %.1f 秒", name, time.monotonic() - started)
        raise
    else:
        logger.info("完成：%s；耗时 %.1f 秒", name, time.monotonic() - started)
    finally:
        stop.set()
        worker.join(timeout=1)
