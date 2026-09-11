"""预设音色朗读 API：提供模型查询、音色查询及 WAV/PCM 语音生成。"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from settings import Settings
from tts_service import TTSService

settings = Settings.from_env()
service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = TTSService(settings)
    try:
        await anyio.to_thread.run_sync(service.load)
        yield
    finally:
        service = None


app = FastAPI(title="Qwen3-TTS CustomVoice API", version="0.3.0", lifespan=lifespan)


def ready_service():
    if service is None or service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return service


@app.get("/health")
async def health():
    current = ready_service()
    return dict(status="ok", model=settings.model_id, dtype=settings.dtype,
                attention=settings.attention, sample_rate=current.sample_rate)


@app.get("/v1/voices")
async def list_voices():
    current = ready_service()
    return dict(voices=current.supported("speakers"), languages=current.supported("languages"))


@app.get("/v1/models")
async def list_models():
    current = ready_service()
    return {"object": "list", "data": [{
        "id": settings.model_id, "object": "model", "owned_by": "Qwen",
        "type": "custom_voice", "available_voices": current.supported("speakers"),
        "available_languages": current.supported("languages"),
    }]}


class ManagedStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # 发送失败或客户端断开时主动关闭异步迭代器，不依赖垃圾回收。
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str = Field(min_length=1, max_length=settings.max_input_chars)
    voice: str = settings.default_voice
    language: str = settings.default_language
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = Field(default=1.0, ge=1.0, le=1.0)
    stream: bool = False
    instruct: Optional[str] = Field(default=None, max_length=settings.max_input_chars)


def canonical_choice(value, choices, label):
    if not choices:
        return value
    lookup = {choice.casefold(): choice for choice in choices}
    if value.casefold() not in lookup:
        raise HTTPException(status_code=400, detail=f"Unsupported {label}: {value}")
    return lookup[value.casefold()]


@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    current = ready_service()
    req.input = req.input.strip()
    if not req.input:
        raise HTTPException(status_code=400, detail="Input text is empty")
    if req.model is not None and req.model != settings.model_id:
        raise HTTPException(status_code=400, detail="Requested model is not loaded")
    # 提前校验音色和语言，避免流式响应发出后才发现参数错误。
    req.voice = canonical_choice(req.voice, current.supported("speakers"), "voice")
    req.language = canonical_choice(req.language, ["Auto", *current.supported("languages")], "language")
    media_type = "audio/pcm" if req.response_format == "pcm" else "audio/wav"
    if req.stream:
        return ManagedStreamingResponse(current.stream(req), media_type=media_type)
    return Response(content=await current.generate(req), media_type=media_type)
