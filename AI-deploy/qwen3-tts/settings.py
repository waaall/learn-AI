"""集中管理部署参数，不在推理和路由中重复维护环境默认值。"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_id: str
    dtype: str
    device: str
    attention: str
    default_voice: str
    default_language: str
    max_input_chars: int
    stream_chunk_size: int

    @classmethod
    def from_env(cls):
        settings = cls(
            model_id=os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            dtype=os.getenv("DTYPE", "bfloat16"),
            device=os.getenv("DEVICE", "cuda:0"),
            attention=os.getenv("ATTN_IMPLEMENTATION", "sdpa"),
            default_voice=os.getenv("DEFAULT_VOICE", "Vivian"),
            default_language=os.getenv("DEFAULT_LANGUAGE", "Auto"),
            max_input_chars=int(os.getenv("MAX_INPUT_CHARS", "2000")),
            stream_chunk_size=int(os.getenv("STREAM_CHUNK_SIZE", "8")),
        )
        if settings.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("DTYPE 必须为 float16、bfloat16 或 float32")
        if settings.max_input_chars <= 0 or settings.stream_chunk_size <= 0:
            raise ValueError("文本长度和流式分块大小必须为正整数")
        return settings
