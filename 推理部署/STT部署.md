
## whisper 部署

- [linuxserver/faster-whisper docker](https://hub.docker.com/r/linuxserver/faster-whisper)

- [whisper-asr-webservice github](https://github.com/ahmetoner/whisper-asr-webservice)
- [whisper-asr-webservice docker](https://hub.docker.com/r/onerahmet/openai-whisper-asr-webservice)
- [whisper-asr-webservice doc](https://ahmetoner.com/whisper-asr-webservice/)

### 下载docker 镜像和源
```bash
# 下载镜像
docker pull onerahmet/openai-whisper-asr-webservice:latest-gpu

# 可以设置国内源
hf download Zoont/faster-whisper-large-v3-turbo-int8-ct2 --local-dir your_path
```

### 启动容器
```yml
services:
  stt:
    image: onerahmet/openai-whisper-asr-webservice:latest-gpu
    container_name: whisper-asr
    ports:
      - "9000:9000"
    environment:
      ASR_ENGINE: "faster_whisper"

      # 把 ASR_MODEL 指向容器内的“模型目录”
      ASR_MODEL: "/models/faster-whisper-large-v3-turbo-int8-ct2"

      # GPU + 量化
      ASR_DEVICE: "cuda"
      ASR_QUANTIZATION: "int8"

      # 可选
      SAMPLE_RATE: "16000"
      MODEL_IDLE_TIMEOUT: "0"  # 0 = 常驻；也可以改成 300 之类让它空闲卸载

    volumes:
      # 把本地下载好的目录挂载到容器内
      - "your-path/faster-whisper-large-v3-turbo-int8-ct2:/models/faster-whisper-large-v3-turbo-int8-ct2:ro"

    restart: unless-stopped

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 访问示例

```bash
curl -X POST "http://192.168.50.50:9000/asr?output=json&language=zh&vad_filter=true&word_timestamps=true" \
  -H "content-type: multipart/form-data" \
  -F "audio_file=@/path/to/audio.wav"

curl -X POST "http://192.168.50.50:9000/asr?output=json&language=zh&vad_filter=true&word_timestamps=true" \
  -H "content-type: multipart/form-data; charset=utf-8" \
  -F "audio_file=@/path/to/audio.wav"
```

## STT处理思路

要“实时性比较好”，核心是别走“上传整段 audio_file 再等返回”的批处理路径，而是做成“边录边传、边出字”。在用 whisper-asr-webservice 的前提下，有两条可落地的路线：

#### 路线A（推荐，体验最好）：前端到后端用 WebSocket 流式，后端接云端实时转写

做法
1. React 前端用 Web Audio API 采集音频，按 20–100ms 切成小块（PCM16 或 Opus）
2. 通过 WebSocket 发给你的 FastAPI “语音网关”
3. 网关把音频流转发给一个支持实时增量转写的 STT（例如 OpenAI Realtime transcription），收到 partial/final 就立刻推回前端
4. final 或端点检测后再触发你的 RAG

优点
- 真正的增量转写（partial / final）
- 延迟低，交互体验最好
- STT 引擎可插拔，将来换本地实现也不影响前端协议

现有的 whisper-asr-webservice 可以继续作为“批处理/离线/兜底 Provider”，但实时主路径用更适合流式的引擎。

#### 路线 B（全部本地、仍尽量实时）：WebSocket 流式 + 你自己做“分段推理”的 STT 代理

因为 whisper-asr-webservice 目前的公开接口是上传文件 /asr 返回结果（批处理），它本身不提供原生流式接口，所以要实时，你需要在它前面加一层“STT 代理服务”（可以还是 FastAPI）来做三件事：

##### 1) 前端流式传音频

- 前端采集音频 → 定时把音频 chunk 推给后端 WS
- 后端把这些 chunk 拼成滚动缓冲区（ring buffer）

##### 2) 端点检测与分段

- 在后端做 VAD（静音检测）
- 每当检测到“句子结束”（静音超过阈值），把这段音频拼成一个临时文件或内存 buffer，调用 whisper-asr-webservice /asr 做一次转写
- 转写出来的结果作为一个 “final segment” 推回前端
- 对于 partial：你可以每隔 0.5–1.0 秒用“最近 N 秒音频”跑一次转写，作为临时 partial（成本高一些，但可用）

##### 3) 合并与去抖

- partial 要做去抖与合并（避免 UI 闪烁）
- final 以“段”为单位追加到文本框
- 端点检测触发 RAG：最后一段 final 产出后，组合整句文本调用 RAG

优点
- 全本地，不依赖云
- 仍可获得“接近实时”的体验（但严格意义上是“准实时”，partial 的代价会更高）

缺点
- partial 做得越频繁越耗 GPU（Whisper 这类模型不是为增量 streaming 设计的）
- 需要你实现 VAD、缓冲、重叠窗口、重复文本对齐这些工程细节

#### 前端怎么做（不写代码的实现要点）

不管走 A 还是 B，前端都建议这样设计：

##### 1) 音频采集与编码

- 采集：AudioWorklet 或 ScriptProcessor（Worklet 更稳）
- 统一格式：优先 16kHz、单声道、PCM16（后端最省事）
- chunk：20ms（320 samples@16k）到 100ms 都可，越小越低延迟但开销更大
- 可选：用 Opus 压缩（省带宽），但后端要解码
  
##### 2) 一个会话一个 WS

- 建立 session_id
- WS 上行：audio_chunk（含 seq、timestamp）
- WS 下行：transcript_partial、transcript_final、error、metrics

##### 3) UI 合并策略

- partial：显示在“灰字临时区”，每次替换而不是追加
- final：追加到“正式文本区”，并清空临时区
- final 触发 RAG：可以“自动触发”或“用户点发送”两种模式
  
#### 端点检测参数建议（决定体感）

- 静音阈值：一般 300–700ms（中文问答场景 500ms 常用）
- 最大段长：10–15 秒（太长会增加延迟）
- 重叠窗口：如果你做 partial 滚动推理，建议保留 1–2 秒 overlap，减少断词
