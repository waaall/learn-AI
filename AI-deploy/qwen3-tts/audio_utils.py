"""编码单声道 PCM16 和 WAV；采样率由模型实际输出提供。"""
import struct

import numpy as np


def to_pcm16(audio) -> bytes:
    return np.clip(np.asarray(audio) * 32768, -32768, 32767).astype("<i2").tobytes()


def wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    # 流式输出无法提前知道总长度，用占位长度；完整输出写入准确长度。
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", riff_size, b"WAVE",
                       b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2,
                       2, 16, b"data", data_len)


def to_wav(audio, sample_rate: int) -> bytes:
    raw = to_pcm16(audio)
    return wav_header(sample_rate, len(raw)) + raw
