## TTS 部署流程

### kokoro 部署流程

可以用python 或者docker ，有作者自己的镜像：

- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)
- [Kokoro-FastAPI-zh](https://github.com/hsiang-han/Kokoro-FastAPI-zh/tree/master)

#### 英文

1. 下载镜像

```bash
# cpu
docker pull ghcr.io/remsky/kokoro-fastapi-cpu:latest
# 国内源 要注意 docker compose 对应也要改
docker pull ghcr.nju.edu.cn/remsky/kokoro-fastapi-cpu:latest

# GPU
docker pull ghcr.io/remsky/kokoro-fastapi-gpu:latest
# 国内源
docker pull ghcr.nju.edu.cn/remsky/kokoro-fastapi-gpu:latest
```

2. docker compose

见：
- [kokoro-tts-compose-gpu.yml](AI-config/whisper-tts-docker/kokoro-tts-compose-gpu.yml)
- [kokoro-tts-compose-cpu.yml](AI-config/whisper-tts-docker/kokoro-tts-compose-cpu.yml)

#### 中文

1. 下载镜像

```bash
# cpu
docker pull ghcr.io/hsiang-han/kokoro-fastapi-zh-cpu:latest
# 国内源 要注意 docker compose 对应也要改
docker pull ghcr.nju.edu.cn/hsiang-han/kokoro-fastapi-zh-cpu:latest

# GPU
docker pull ghcr.io/hsiang-han/kokoro-fastapi-zh-gpu:latest
# 国内源 要注意 docker compose 对应也要改
docker pull ghcr.nju.edu.cn/hsiang-han/kokoro-fastapi-zh-gpu:latest
```


## 现代 TTS 四模型对比：XTTS-v2、Kokoro、Qwen3-TTS、OmniVoice

|模型|当前热度|发布与状态|技术特点|许可证|
|---|---|---|---|---|
|Coqui XTTS-v2|HF 近月约 8.57M 下载，老牌高热模型|2023 发布，模型卡最后更新较早；生态成熟但官方活跃度不如新模型|支持 17 种语言；约 6 秒参考音频即可跨语言 voice cloning；24 kHz；支持情绪/风格迁移；工程资料多|CPML，偏非商业限制|
|Kokoro-82M|HF 近月约 9.7M 下载，当前最热门轻量 TTS 之一|2024 末/2025 初爆发；2026 仍是轻量本地部署代表|82M 参数；StyleTTS2 + iSTFTNet；速度快、成本低；多语言、多音色；不主打任意 voice cloning|Apache-licensed weights，商业友好|
|Qwen3-TTS|HF 1.7B CustomVoice 近月约 1.52M 下载；GitHub 约 11k+ stars|2026-01 开源；Qwen 官方 TTS 系列|基于 500万小时+ 语音数据；支持 10 种语言；3 秒 voice cloning；description-based voice control；支持 streaming；有 0.6B/1.7B 等规格|Apache-2.0，商业友好|
|OmniVoice|HF 近月约 2.09M 下载；发布后增长很快|2026-04 发布；k2-fsa 开源模型|支持 600+ / 646 种语言；zero-shot voice cloning；voice design；支持 `[laughter]` 等非语言符号和拼音/音素纠错；Diffusion Language Model-style 架构；标称 RTF 低至 0.025|Apache-2.0，商业友好|

### 逐个简评

XTTS-v2：成熟的“老兵型”语音克隆模型。
它的核心优势是生态成熟、教程多、推理链路比较稳定，仍然适合做多语言 voice cloning 原型。模型卡明确写着支持 17 种语言、6 秒音频克隆、跨语言克隆、24 kHz 输出等特性。问题在于许可证是 Coqui Public Model License（CPML），商业落地会很尴尬；同时模型本身更新较早，新一代模型在可控性、许可证和多语言覆盖上已经反超。

Kokoro：轻量部署之王。
Kokoro 的定位和另外三个不太一样，它不是“克隆任意人的声音”的模型，而是“用很小成本稳定生成好听语音”的模型。82M 参数、Apache 许可权重、可商用、推理便宜，这些让它非常适合做 API、离线助手、阅读器、游戏 NPC 语音、移动端或边缘侧 TTS。模型卡也明确强调它虽然轻量，但质量接近更大模型，并且部署成本很低。

Qwen3-TTS：中文/英文产品化语音 Agent 的强候选。
Qwen3-TTS 更像“现代语音 Agent TTS”：不仅能克隆，还能用自然语言描述声音，比如音色、风格、表达方式。官方模型卡写明它基于 500 万小时以上语音数据，覆盖 10 种语言，支持 3 秒 voice cloning 和 description-based control，并且强调 streaming。它比 Kokoro 重，但比 XTTS-v2 更适合商业产品，因为许可证是 Apache-2.0。

OmniVoice：新一代“语言覆盖怪兽”。
OmniVoice 最大的卖点是 600+ 语言，HF 页面甚至标注为 646 languages。它不只是多语言，还支持 zero-shot voice cloning、voice design、非语言符号控制、拼音/音素纠错，并使用 Diffusion Language Model-style 架构。官方标称 RTF 可低至 0.025，也就是最快可达 40 倍实时。它很新，生态成熟度还不如 Kokoro/XTTS，但如果的项目涉及低资源语言、多语种覆盖、全球化语音合成，它是这四个里最值得重点测试的新模型。

### 关键差异

|维度|最强/最合适|
|---|---|
|轻量本地部署|Kokoro|
|成熟 voice cloning 生态|XTTS-v2|
|商业友好 voice cloning|Qwen3-TTS / OmniVoice|
|自然语言声音设计|Qwen3-TTS / OmniVoice|
|超多语言覆盖|OmniVoice|
|中文/英文语音 Agent|Qwen3-TTS|
|低成本 TTS API|Kokoro|
|老项目兼容与教程数量|XTTS-v2|

### 选型建议

如果要做商业产品，优先排除 XTTS-v2，除非只是内部研究或能解决授权问题。它技术仍然能打，但许可证像一把卡在门缝里的钥匙，能开实验室门，不一定能开公司大门。

如果要做低成本、高并发、本地或边缘部署，选 Kokoro。它不是功能最花哨的，但部署体验最朴素、最省心，像一把小而锋利的折刀。

如果要做中文/英文语音助手、AI Agent、角色语音、可控语音生成，选 Qwen3-TTS。它在克隆、声音描述、流式生成和商业许可之间平衡得很好。

如果要做多语言、低资源语言、国际化语音生成，重点测 OmniVoice。600+ 语言覆盖是它的压舱石，也是目前这四个里最突出的差异点。

### 简短结论

Kokoro 适合“便宜、快、稳”；Qwen3-TTS 适合“可商用、可控、适合 Agent”；OmniVoice 适合“超多语言 + 新一代克隆”；XTTS-v2 适合“成熟克隆生态和非商业原型”。

我的排序建议：

1. 产品落地优先：Qwen3-TTS、Kokoro
2. 多语言覆盖优先：OmniVoice
3. 实验/旧项目兼容：XTTS-v2
4. 商业化谨慎使用：XTTS-v2