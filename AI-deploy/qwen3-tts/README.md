# Qwen3-TTS：4090 预设音色朗读

这是独立的 CustomVoice 部署目录。仅提供预设音色朗读，不提供声音克隆、VoiceDesign 或请求时动态切换模型。

## 服务器部署

把**整个目录（包括 scripts 和隐藏配置文件）**复制到 GPU 服务器。下面命令均在服务器的部署目录执行：

```bash
# 创建机器专属配置，随后编辑模型缓存目录、端口、GPU 编号等。
cp .env.example .env
nvidia-smi

# 确认服务器已有的镜像以及容器 GPU 可用性，不自动拉取替代镜像。
docker image inspect pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
docker run --rm --pull=never --gpus all \
  pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime \
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()'

# 使用服务器默认 Docker builder；独立 buildx builder 未必能看到本地镜像。
docker compose -f qwen3-tts-compose-gpu.yml config
docker compose -f qwen3-tts-compose-gpu.yml build --pull=false
docker compose -f qwen3-tts-compose-gpu.yml up -d --no-build --pull never
docker compose -f qwen3-tts-compose-gpu.yml logs -f
```

宿主机须已安装 NVIDIA 驱动和 NVIDIA Container Toolkit。无需因容器使用 CUDA 12.8 就在宿主机另装同版本 CUDA Toolkit，但驱动必须兼容容器运行时。

默认宿主机端口 18765，容器内部仍使用 8080，监听所有网卡。**没有 API 鉴权，仅适合可信局域网；不要直接映射到公网。** 可用 `BIND_ADDRESS` 限定服务器 LAN 地址，并用防火墙限制访问。

## 镜像与下载源

- 默认复用 `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`，使用 BF16 + SDPA，不安装 FlashAttention。
- `BASE_IMAGE`、`APT_MIRROR`、`APT_SECURITY_MIRROR`、`PIP_INDEX_URL` 属于构建参数，更改后需重新构建。
- apt 默认连安全更新也使用清华源，避免国内网络卡在官方安全源。镜像可能有同步延迟；可将 `APT_SECURITY_MIRROR` 改为 `https://security.ubuntu.com/ubuntu`。
- 换 apt 源保留基础镜像的发行版及签名配置，不按宿主机发行版重写。
- pip 使用单一 HTTPS 国内源，不关闭证书校验。未配置多个索引作为自动回退。
- `HF_ENDPOINT` 是运行时模型下载端点，默认 HF-Mirror；可切回 `https://huggingface.co`。第三方镜像不保证所有文件均可访问，不要向其提供私有模型凭据。
- 模型默认缓存在服务器部署目录的 `models` 中。可将 `.env` 中 `MODEL_CACHE_DIR` 改为宿主机绝对路径。
- 模型不会打包进镜像。首次启动会下载，健康检查宽限期为 30 分钟；健康状态不会让 Docker 自动重启容器。
- 完成所有模型及 tokenizer 下载后，可设置 `HF_HUB_OFFLINE=1` 再重建容器，验证完整离线运行。
- Docker Hub/GHCR 镜像拉取不受 apt、pip、HF 源影响；`--pull=false` 不是完全离线构建保证，构建器仍可能访问镜像元数据。

## 依赖与验证边界

直接依赖固定在 `requirements.txt`，其中 `faster-qwen3-tts==0.4.0` 的 PyPI 元数据声明 `torch>=2.5.1`，并依赖 `qwen-tts-hf` 和 Transformers 5.x。基础镜像中的 torch/torchvision/torchaudio/triton（已安装部分）会生成版本约束，冲突则构建失败，不自动换掉原有 PyTorch。构建末尾执行 `pip check` 和导入检查。

**这不是完整的传递依赖锁文件，也不是已经通过 4090 实机测试的组合。** 初次构建和推理成功后，可保存 `pip freeze` 作为该服务器的依赖快照。不要遇到冲突就删掉 PyTorch 约束；应先检查冲突包。如果运行时需要编译 CUDA 扩展，再考虑带开发工具链的基础镜像。

参考：[固定版本元数据](https://pypi.org/pypi/faster-qwen3-tts/0.4.0/json)、[上游接口](https://github.com/andimarafioti/faster-qwen3-tts/blob/main/faster_qwen3_tts/model.py)、[清华 apt](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)、[清华 pip](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)、[HF-Mirror](https://hf-mirror.com/)。

## 调用

```bash
# 将地址改为 GPU 服务器的局域网地址和映射端口。
API_URL=http://127.0.0.1:18765
curl -f "$API_URL/health"
curl -f "$API_URL/v1/voices"

# 音色名称以 /v1/voices 返回内容为准，默认 Vivian。
curl --fail-with-body "$API_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，这是预设音色朗读测试。","voice":"Vivian","language":"Chinese","response_format":"wav"}' \
  --output speech.wav
```

支持 `response_format=wav|pcm`、`stream=true|false`、可选 `instruct`。`speed` 暂仅接受 `1.0`，其他值明确拒绝；不支持的格式不会偷偷返回 WAV。`model` 可省略，提供时必须等于当前加载模型 ID。音色、语言从模型读取，匹配时不区分大小写。

PCM 为单声道、小端有符号 16 位，采样率见 `/health`。流式 WAV 使用未知长度头，部分播放器不兼容；需要精确文件长度时使用非流式 WAV。

模型推理串行执行，不做并行批处理；阻塞推理在线程池中运行，避免卡住健康检查。流式客户端断开时会关闭生成器并释放模型锁，但正在执行的单次 GPU 调用不能瞬间中止。

## 本地无 GPU 测试

测试使用替身模型，不代表真实推理兼容性。安装应用依赖及 pytest/httpx 后，在部署目录执行：

```bash
python -m pytest tests -q
bash -n entrypoint.sh
```
