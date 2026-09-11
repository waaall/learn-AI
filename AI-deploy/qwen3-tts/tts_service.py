"""封装预设音色推理，串行使用模型，避免阻塞服务事件循环。"""
from contextlib import closing
from functools import partial
import logging
import os

import anyio

from audio_utils import to_pcm16, to_wav, wav_header
from startup_logging import startup_stage

logger = logging.getLogger("qwen3_tts.service")


class TTSService:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.lock = anyio.Lock()

    def load(self):
        model_path = os.getenv("MODEL_LOAD_PATH", "").strip()
        if not model_path:
            raise RuntimeError("模型尚未准备，请通过 entrypoint.sh 启动服务")
        with startup_stage("导入 PyTorch 和 TTS 依赖"):
            import torch
            from faster_qwen3_tts import FasterQwen3TTS

        with startup_stage("检查 CUDA 设备"):
            logger.info("PyTorch=%s，CUDA runtime=%s，device=%s，dtype=%s，attention=%s",
                        torch.__version__, torch.version.cuda, self.settings.device,
                        self.settings.dtype, self.settings.attention)
            if not self.settings.device.startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError("此服务需要可用的 CUDA GPU，请检查驱动和容器 GPU 配置")
            device = torch.device(self.settings.device)
            free, total = torch.cuda.mem_get_info(device)
            logger.info("GPU=%s，空闲显存=%.2f GiB，总显存=%.2f GiB",
                        torch.cuda.get_device_name(device), free / 1024 ** 3, total / 1024 ** 3)

        logger.info("准备从本地加载模型=%s；HF_HOME=%s；离线模式=%s",
                    model_path, os.getenv("HF_HOME", "默认目录"),
                    os.getenv("HF_HUB_OFFLINE", "0"))
        logger.info("模型文件准备完成；加载完成前 HTTP 端口尚未监听。")
        # 下载已经独立完成；GPU 初始化错误不触发重新下载或换源。
        with startup_stage("离线加载权重及初始化 CUDA Graph"):
            self.model = FasterQwen3TTS.from_pretrained(
                model_path, device=self.settings.device,
                dtype=getattr(torch, self.settings.dtype),
                attn_implementation=self.settings.attention,
            )
            if getattr(self.model.model.model, "tts_model_type", None) != "custom_voice":
                raise ValueError("此服务仅部署 CustomVoice 预设音色模型")
        logger.info("模型已加载：采样率=%s Hz，音色=%s", self.sample_rate, self.supported("speakers"))
        logger.info("应用初始化完成，即将由 Uvicorn 开始监听容器端口 %s；首次合成仍可能触发预热。",
                    os.getenv("PORT", "8080"))

    @property
    def sample_rate(self):
        return self.model.sample_rate if self.model is not None else None

    def supported(self, name):
        if self.model is None:
            return []
        # 优先使用包装器公开接口，兼容底层实现提供同名方法的情况。
        for obj in (self.model.model, self.model.model.model):
            fn = getattr(obj, f"get_supported_{name}", None)
            if callable(fn):
                return [str(value) for value in (fn() or [])]
        return []

    def arguments(self, req):
        return dict(text=req.input, speaker=req.voice, language=req.language,
                    instruct=req.instruct or None)

    async def generate(self, req):
        async with self.lock:
            # 在工作线程执行 GPU 调用及编码，健康检查仍可及时响应。
            return await anyio.to_thread.run_sync(partial(self._generate, req))

    def _generate(self, req):
        wavs, sr = self.model.generate_custom_voice(**self.arguments(req))
        return to_pcm16(wavs[0]) if req.response_format == "pcm" else to_wav(wavs[0], sr)

    def _chunks(self, req):
        source = self.model.generate_custom_voice_streaming(
            **self.arguments(req), chunk_size=self.settings.stream_chunk_size,
        )
        with closing(source):
            header_sent = False
            for chunk, sr, _timing in source:
                if req.response_format == "wav" and not header_sent:
                    yield wav_header(sr)
                    header_sent = True
                yield to_pcm16(chunk)

    async def stream(self, req):
        async with self.lock:
            chunks = self._chunks(req)
            sentinel = object()
            try:
                while True:
                    # 不让 StopIteration 穿过异步 Future 边界。
                    chunk = await anyio.to_thread.run_sync(lambda: next(chunks, sentinel))
                    if chunk is sentinel:
                        break
                    yield chunk
            finally:
                # 客户端断开或推理异常时也关闭底层生成器，再释放模型锁。
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(chunks.close)
