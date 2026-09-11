"""用替身模型检查接口、音频和并发行为，不下载权重或调用 GPU。"""
import asyncio
import io
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from audio_utils import to_pcm16, to_wav
from tts_service import TTSService
from scripts.configure_apt import configure


class FakeModel:
    sample_rate = 24000

    def __init__(self):
        self.closed = False
        self.model = SimpleNamespace(
            get_supported_speakers=lambda: ["Vivian", "Ryan"],
            get_supported_languages=lambda: ["Chinese", "English"],
            model=SimpleNamespace(tts_model_type="custom_voice"),
        )

    def generate_custom_voice(self, **kwargs):
        return [np.array([0, 0.5, -0.5])], self.sample_rate

    def generate_custom_voice_streaming(self, **kwargs):
        try:
            yield np.array([0, 0.5]), self.sample_rate, {}
            yield np.array([-0.5]), self.sample_rate, {}
        finally:
            self.closed = True


@pytest.fixture
def client(monkeypatch):
    def load(service):
        service.model = FakeModel()
    monkeypatch.setattr(TTSService, "load", load)
    with TestClient(main.app) as client:
        yield client


def test_audio_encoding():
    with wave.open(io.BytesIO(to_wav(np.array([0, 2, -2]), 24000))) as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 24000
        assert audio.getnframes() == 3
        assert audio.readframes(3) == b"\x00\x00\xff\x7f\x00\x80"


def test_queries(client):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/voices").json()["voices"] == ["Vivian", "Ryan"]
    assert client.get("/v1/models").json()["data"][0]["type"] == "custom_voice"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fmt", ["wav", "pcm"])
def test_speech(client, stream, fmt):
    response = client.post("/v1/audio/speech", json={
        "input": "你好", "voice": "vivian", "language": "chinese",
        "stream": stream, "response_format": fmt,
    })
    assert response.status_code == 200
    assert response.headers["content-type"] == f"audio/{fmt}"
    pcm = to_pcm16(np.array([0, 0.5, -0.5]))
    assert (response.content[44:] if fmt == "wav" else response.content) == pcm
    if stream:
        assert main.service.model.closed


@pytest.mark.parametrize("extra,status", [
    ({"input": " "}, 400), ({"input": "x" * 2001}, 422),
    ({"response_format": "mp3"}, 422), ({"speed": 1.2}, 422),
    ({"voice": "unknown"}, 400), ({"language": "unknown"}, 400),
    ({"model": "other"}, 400),
])
def test_validation(client, extra, status):
    assert client.post("/v1/audio/speech", json={"input": "你好", **extra}).status_code == status


def test_unready(monkeypatch):
    monkeypatch.setattr(main, "service", None)
    response = TestClient(main.app).get("/health")
    assert response.status_code == 503


def test_health_during_generation(monkeypatch):
    async def scenario():
        service = TTSService(main.settings)
        service.model = FakeModel()
        entered, release = threading.Event(), threading.Event()
        original = service.model.generate_custom_voice

        def slow_generate(**kwargs):
            entered.set()
            assert release.wait(5)
            return original(**kwargs)

        service.model.generate_custom_voice = slow_generate
        monkeypatch.setattr(main, "service", service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            task = asyncio.create_task(client.post("/v1/audio/speech", json={"input": "你好"}))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                response = await asyncio.wait_for(client.get("/health"), 1)
                assert response.status_code == 200
            finally:
                release.set()
                assert (await task).status_code == 200
    asyncio.run(scenario())


def test_close_stream_releases_model():
    async def scenario():
        service = TTSService(main.settings)
        service.model = FakeModel()
        stream = service.stream(main.SpeechRequest(input="你好"))
        await anext(stream)
        await stream.aclose()
        assert service.model.closed
        assert not service.lock.locked()
    asyncio.run(scenario())


def test_apt_formats(tmp_path):
    sources = tmp_path / "sources.list.d"
    sources.mkdir()
    legacy = tmp_path / "sources.list"
    legacy.write_text("deb http://archive.ubuntu.com/ubuntu jammy main\n")
    modern = sources / "ubuntu.sources"
    modern.write_text("URIs: http://security.ubuntu.com/ubuntu\nSuites: noble-security\nSigned-By: /key\n")
    nvidia = sources / "cuda.list"
    nvidia.write_text("deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /\n")
    original = nvidia.read_text()
    configure(tmp_path, "https://mirror.example/ubuntu", "https://security.example/ubuntu")
    assert "https://mirror.example/ubuntu/ jammy" in legacy.read_text()
    assert "https://security.example/ubuntu/" in modern.read_text()
    assert "Suites: noble-security\nSigned-By: /key" in modern.read_text()
    assert nvidia.read_text() == original


def test_stream_send_failure_closes_model():
    async def scenario():
        service = TTSService(main.settings)
        service.model = FakeModel()
        response = main.ManagedStreamingResponse(service.stream(main.SpeechRequest(input="你好")))

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("客户端已断开")

        with pytest.raises(Exception):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert service.model.closed
        assert not service.lock.locked()
    asyncio.run(scenario())
