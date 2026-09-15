# Qwen3-TTS：4090 预设音色朗读

这是独立的 CustomVoice 部署目录。仅提供预设音色朗读，不提供声音克隆、VoiceDesign 或请求时动态切换模型。

## 服务器部署

把**整个目录（包括 scripts 和隐藏配置文件，但不包含凭据）**复制到 GPU 服务器。下面命令均在服务器的部署目录执行：

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
- 模型不会打包进镜像。首次启动先准备模型文件再启动 API，健康检查宽限期为 30 分钟；健康状态不会让 Docker 自动重启容器。下载耗时可能超过宽限期，届时显示 unhealthy 不等于进程已退出。
- 完成所有模型及 tokenizer 下载后，可设置 `HF_HUB_OFFLINE=1` 再重建容器，验证完整离线运行。
- Docker Hub/GHCR 镜像拉取不受 apt、pip、HF 源影响；`--pull=false` 不是完全离线构建保证，构建器仍可能访问镜像元数据。

## 依赖与验证边界

直接依赖固定在 `requirements.txt`，当前使用 `faster-qwen3-tts==0.4.0`，并额外锁定 `transformers==5.15.1`。该 faster 版本依赖 `qwen-tts-hf>=0.1.1.post1,<0.2`、`transformers>=5.15.1,<6`、`huggingface-hub>=1.5.0,<2.0` 和 `torch>=2.5.1`；其他间接依赖仍由 pip 在上游约束范围内解析。基础镜像中的 torch/torchvision/torchaudio/triton（已安装部分）会生成版本约束，冲突则构建失败，不自动换掉原有 PyTorch。构建末尾执行 `pip check` 和导入检查。

本次锁定用于验证原环境 `transformers==5.17.0` 下的 `MimiConfig.rope_theta` 初始化错误是否与 Transformers 版本有关，尚不能认定 `5.15.1` 已修复问题。应重新构建镜像，不在旧容器中覆盖安装，也不混装 `qwen-tts` 与 `qwen-tts-hf`；保留挂载的模型缓存。不修改依赖库源码，不增加 pip 自动换源机制。仍须在目标 GPU 上验证完整模型加载、普通合成和流式合成，成功后再记录完整依赖版本；构建检查通过不代表推理已验证。

**这不是完整的传递依赖锁文件，也不是已经通过 4090 实机测试的组合。** 初次构建和推理成功后，可保存 `pip freeze` 作为该服务器的依赖快照。不要遇到冲突就删掉 PyTorch 约束；应先检查冲突包。如果运行时需要编译 CUDA 扩展，再考虑带开发工具链的基础镜像。

参考：[固定版本依赖声明](https://github.com/andimarafioti/faster-qwen3-tts/blob/v0.4.0/pyproject.toml)、[上游接口](https://github.com/andimarafioti/faster-qwen3-tts/blob/v0.4.0/faster_qwen3_tts/model.py)、[清华 apt](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)、[清华 pip](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)、[HF-Mirror](https://hf-mirror.com/)。

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

## 查看启动进度

`docker compose -f qwen3-tts-compose-gpu.yml logs -f --tail=100` 可看到模型文件准备、换源重试、依赖导入、CUDA 检查、模型初始化和就绪日志；下载错误仅输出异常类型及 HTTP 状态码，避免暴露凭据或签名 URL，其他启动失败输出异常堆栈。默认每 30 秒输出当前阶段耗时、Hub 缓存总大小和 `.incomplete` 文件数量，可通过 `STARTUP_LOG_INTERVAL` 调整。

缓存统计包含已有模型，临时文件也可能来自上次中断，因此不是当前模型的下载百分比。状态日志持续输出也不保证下载或加载正在推进；需结合缓存增长、上游日志及 GPU 状态判断。模型初始化期间 HTTP 尚未开始监听，出现 Uvicorn 监听日志后再访问 `/health` 或 `/docs`。

此变更需要重新构建镜像并重建容器，单纯 `docker restart` 不会更新代码。

## 模型来源与自动回退

### 方案一：自动下载（默认）

只使用主 Compose 即可，无需预先确认官方源能否直连。处理顺序：

1. 若启用了外部目录覆盖配置，直接校验并使用该目录，不联网。
2. 否则检查已准备好的 snapshot 和 Hub 缓存；具备所需运行文件则直接使用。
3. 缺文件时先尝试 `HF_ENDPOINT`（默认 HF-Mirror），每个源最多 `MODEL_DOWNLOAD_ATTEMPTS=2` 次。
4. 网络、HTTP（包括 401）或 Xet 下载错误导致尝试失败后，自动回退到 `https://huggingface.co`。`HF_FALLBACK_TO_OFFICIAL=0` 可关闭回退。
5. 每次下载进程最多 `MODEL_DOWNLOAD_TIMEOUT=900` 秒，超时终止并保留缓存。`HF_HUB_DOWNLOAD_TIMEOUT=30` 是请求读取超时，不是整个模型下载时限。
6. 全部失败则退出，不会进入 GPU 模型加载。默认容器重启策略改为 `on-failure:3`，避免原先 `unless-stopped` 无限制重启；网络恢复后可手动启动。此策略与 `unless-stopped` 的宿主机重启行为不同，容器不会仅因 Docker daemon 重启自动恢复。

`HF_HUB_DISABLE_XET=1` 默认禁用 Xet 客户端。这不是保证下载完全避开所有 Xet/CDN 域名，也不能保证官方源一定可达。

首次获取模型元数据后，将 revision 固定为提交哈希，之后重试或换源都沿用该提交。如果已有部分缓存，也尽量沿用其提交。完整运行文件会保存在原来的 Hub 缓存，记录准备结果的 `.qwen-prepared` 目录同样位于缓存内，不复制另一份权重。不会为了重试主动清空缓存或强制重下全部权重；未完成的大文件是否能续传取决于下载库和服务端。

**`main` 默认优先使用已缓存版本，不自动追踪模型更新。** 要更新模型，建议把 `MODEL_REVISION` 设为目标提交哈希。跨镜像不能仅凭仓库名称确认内容可信，应使用可信来源；本地检查只验证必需文件、分片存在性及 LFS 指针，不是完整的权重哈希校验。

模型准备完成后，以本地 snapshot 路径启动新 API 进程，并设置 Hub/Transformers 离线模式。权重加载或显存不足不触发换源。

### 方案二：可选的官方 Token

401 并不一定是缺少个人 Token。需要时，在**服务器**创建只包含 Token 文本的文件（建议最小读取权限的 Token，文件权限 `600`），不要把 Token 发到聊天、写进 Dockerfile 或提交到 Git。

#### Unix / Linux / macOS / WSL / Git Bash

以下命令在**实际运行 Docker Engine 的主机**上执行，并且当前目录应为本部署目录：

```bash
cd /opt/qwen3-tts                 # 按实际部署目录修改
test -f .env || cp .env.example .env
mkdir -p secrets
vim secrets/hf-token
```

在 Vim 中按 `i` 开始输入，粘贴一行 `hf_...` Token；按 `Esc`，输入 `:wq`，再按回车保存。Token 文件允许末尾有一个换行，但不要有空格、引号或其他内容：

```bash
chmod 600 secrets/hf-token
```

如果本机没有 Vim，也可以用 `nano secrets/hf-token`。不要使用 `cat`、`echo` 或调试命令把 Token 打到终端或日志中。

#### Windows PowerShell

如果 Docker Desktop 运行在 Windows 本机，先把 Docker Desktop 切换为 **Linux containers**，并启用 WSL2 后端；GPU 容器需要 WSL2。若 Windows 只是远程操作 Linux 4090 服务器，应改用上面的 Unix 步骤，Token 文件必须位于 Linux 服务器，而不是 Windows 客户端。

```powershell
Set-Location C:\AI\qwen3-tts       # 按实际部署目录修改
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
New-Item -ItemType Directory -Force .\secrets | Out-Null
notepad .\secrets\hf-token
```

在记事本中粘贴一行 `hf_...` Token 后保存。文件名必须是 `hf-token`，不是 `hf-token.txt`；编码选择 `UTF-8`（不要保存成 UTF-16），内容不要包含空格、引号或其他行。也可以在 WSL / Git Bash 中直接使用上面的 `vim secrets/hf-token`。

Windows 下 `.env` 的宿主机路径建议使用相对路径，避免反斜杠和盘符解析问题：

```dotenv
HF_TOKEN_HOST_FILE=./secrets/hf-token
HF_ENDPOINT=https://hf-mirror.com
HF_FALLBACK_TO_OFFICIAL=1
```

若必须使用绝对路径，使用正斜杠，例如 `C:/AI/qwen3-tts/secrets/hf-token`，不要写成容器内的 `/run/secrets/hf_token`。后者是 Compose 在容器内自动挂载后的路径，不能作为宿主机文件路径。

合并凭据覆盖文件启动：

```bash
docker compose --env-file .env -f qwen3-tts-compose-gpu.yml -f qwen3-tts-token.override.yml config --quiet
docker compose --env-file .env -f qwen3-tts-compose-gpu.yml build --pull=false
docker compose --env-file .env -f qwen3-tts-compose-gpu.yml -f qwen3-tts-token.override.yml up -d --no-build --pull never --force-recreate
docker compose --env-file .env -f qwen3-tts-compose-gpu.yml -f qwen3-tts-token.override.yml logs -f --tail=100 qwen3-tts
```

`config --quiet` 只检查合并后的配置，不会显示 Token 内容。若已有最新镜像，第二条 `build` 可跳过；新增或更新 `prepare_model.py`、下载回退逻辑后必须重新构建。只新增 Token 文件时仍要 `--force-recreate`，因为 Secret 挂载属于容器配置。

Token 通过 Compose secret 文件只读挂载，仅在下载端点**精确等于 `https://huggingface.co`** 时显式传给官方 Hub 客户端；镜像请求使用 `token=False`。启动脚本会忽略普通 `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` 环境变量，避免自动把凭据发给镜像。应用不会把 Token 保存到模型缓存。

注意：本地 Compose secret 是只读文件挂载，不是加密凭据仓库。`secrets/` 已被 Git 和 Docker 构建上下文忽略，任意其他路径的凭据仍需自行保管。不要配置带用户名密码或 Token 查询参数的 `HF_ENDPOINT`。

### 方案三：只读加载宿主机完整模型目录

在 `.env` 中指定：

Unix / Linux / WSL：

```dotenv
HOST_MODEL_DIR=/absolute/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Windows Docker Desktop：

```dotenv
HOST_MODEL_DIR=C:/AI/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

合并本地模型覆盖文件启动：

```bash
docker compose --env-file .env -f qwen3-tts-compose-gpu.yml -f qwen3-tts-local.override.yml up -d --no-build --pull never --force-recreate
```

Windows PowerShell 也执行同一条命令；如果当前目录不是部署目录，请先 `Set-Location C:\AI\qwen3-tts`。本地模型模式不需要 Token，且 `HF_HUB_OFFLINE=1` 会阻止联网；不要同时误用 Token 覆盖文件来排查下载问题。

目录挂载到容器 `/models/customvoice`，只读，路径不存在时不自动创建空目录。目录必须包含模型配置、生成配置、预处理配置、文本 tokenizer、主模型权重，以及完整的 `speech_tokenizer/` 子目录；分片权重必须包含索引列出的所有分片。不支持只挂一个 `.safetensors` 文件。`MODEL_ID` 仍作为 API 的模型名称，不暴露本地路径。

缺文件、Git LFS 指针文件或损坏的 snapshot 软链接会明确报错，不会偷偷回退联网。若复制的是 Hugging Face snapshot 目录，应连同其引用的 blobs 一起保留目录关系，或复制为真实文件，避免链接失效。

上述覆盖文件仅用于与主 Compose 合并，日志限制等设置继承主文件。每次更新或重建容器时须使用相同的 `-f` 文件组合。首次部署这些新代码仍需先执行主 Compose 的 `build --pull=false`；已有 `.env` 不要用示例覆盖，只添加所需新选项。

接口或推理功能未在 Mac 上运行验证；新增的文件准备逻辑需要在目标服务器验证。可在服务器运行 `python -m pytest tests/test_model_sources.py -q` 做无下载、无 GPU 的替身测试，再验证真实下载及模型加载。
