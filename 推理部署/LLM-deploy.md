
# LLM 部署平台

vllm、llama.cpp、ollama、openllm

vllm是支持并发最好的，llama.cpp是支持平台最多的，ollama是最简单性能也是最差的。

## [vllm](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)Nvidia部署

```
┌─────────────────────────────────────────────┐
│           vllm/vllm-openai 容器              │
│  ┌───────────────────────────────────────┐  │
│  │  CUDA Runtime (cuDNN, cuBLAS等)        │  │  ← 镜像自带
│  │  PyTorch, vLLM, Python                │  │
│  └───────────────────────────────────────┘  │
├─────────────────────────────────────────────┤
│        nvidia-container-toolkit             │  ← 宿主机安装
├─────────────────────────────────────────────┤
│           NVIDIA Driver                     │  ← 宿主机安装
├─────────────────────────────────────────────┤
│              GPU 硬件                        │
└─────────────────────────────────────────────┘
```


- linux 需要先安装显卡驱动 和 nvidia-container-toolkit。比如ubuntu可以通过apt安装，但一般都会比较老，会有兼容性问题，去英伟达官方搜索。
- windows需要安装nvidia驱动和wsl2(不建议在windows部署)

### 查看vllm模型参数/状态/性能


#### 1. 查看 vLLM 运行时参数

##### 方法一：API 端点查询

vLLM 提供了多个 API 端点可以查看运行时信息：

```bash
# 基本模型信息
curl http://localhost:8123/v1/models | jq

# 详细配置信息（vLLM 特有）
curl http://localhost:8123/v1/model_info | jq

# 服务器健康检查
curl http://localhost:8123/health
```

##### 方法二：进入容器查看日志

```bash
# 查看启动日志（包含详细配置）
docker logs vllm-qwen3-4090-awq

# 实时跟踪日志
docker logs -f vllm-qwen3-4090-awq
```

启动日志会显示类似这样的关键信息：

```
INFO: Model config: ...
INFO: KV cache data type: auto
INFO: GPU memory utilization: 0.90
INFO: Maximum number of batched tokens: 8192
INFO: Number of GPU blocks: XXXX
INFO: Number of CPU blocks: XXXX
```

##### 方法三：Python 脚本查询详细信息

```python
import requests
import json

BASE_URL = "http://localhost:8123"

def get_model_info():
    """获取模型基本信息"""
    resp = requests.get(f"{BASE_URL}/v1/models")
    print("=== 模型列表 ===")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

def get_detailed_info():
    """获取详细配置（vLLM 特有端点）"""
    endpoints = [
        "/v1/model_info",
        "/metrics",  # Prometheus 格式的指标
    ]

    for ep in endpoints:
        try:
            resp = requests.get(f"{BASE_URL}{ep}")
            print(f"\n=== {ep} ===")
            if "json" in resp.headers.get("content-type", ""):
                print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
            else:
                # metrics 是文本格式
                print(resp.text[:2000])  # 截断显示
        except Exception as e:
            print(f"{ep}: {e}")

def get_metrics_parsed():
    """解析 Prometheus 指标中的 KV Cache 信息"""
    resp = requests.get(f"{BASE_URL}/metrics")
    lines = resp.text.split('\n')

    print("\n=== KV Cache 相关指标 ===")
    kv_keywords = ['kv_cache', 'gpu_cache', 'cache_block', 'prefix_cache']
    for line in lines:
        if any(kw in line.lower() for kw in kv_keywords):
            print(line)

    print("\n=== GPU 内存相关 ===")
    mem_keywords = ['gpu_memory', 'memory_usage']
    for line in lines:
        if any(kw in line.lower() for kw in mem_keywords):
            print(line)

if __name__ == "__main__":
    get_model_info()
    get_detailed_info()
    get_metrics_parsed()
```

##### 方法四：进入容器执行诊断

```bash
# 进入容器
docker exec -it vllm-qwen3-4090-awq bash

# 在容器内查看 GPU 状态
nvidia-smi

# 查看 Python 环境中的 vLLM 配置
python -c "import vllm; print(vllm.__version__)"
```

---

#### 2. vLLM Benchmark 测试

##### vllm bench 的部署方式

**两种方式都可以**：

- **本地 Python**：直接 pip install vllm 后使用
- **Docker 内执行**：进入已有容器或启动新容器

##### 方式一：本地 Python 安装（推荐用于 benchmark）

```bash
# 创建虚拟环境
conda create -n vllm-bench python=3.11 -y
conda activate vllm-bench

# 安装 vllm（与你的 Docker 版本一致）
pip install vllm==0.12.0
```

##### 方式二：在 Docker 容器内执行

```bash
# 进入正在运行的容器
docker exec -it vllm-qwen3-4090-awq bash

# 或者启动新容器专门做 benchmark
docker run --rm -it --gpus all \
    -v "D:/dev_software/AI_models/huggingface/Qwen3-30B-A3B-AWQ-4bit:/models/qwen3-awq:ro" \
    vllm/vllm-openai:v0.12.0 \
    bash
```

---

#### 3. Benchmark 命令详解

##### 3.1 离线 Throughput 测试（不需要服务运行）

```bash
# 在容器内或本地 Python 环境执行
vllm bench throughput \
    --model /models/qwen3-awq \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 100 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --quantization awq
```

##### 3.2 在线延迟测试（需要服务运行）

先确保你的 Docker 服务已启动，然后：

```bash
# 测试延迟
vllm bench latency \
    --model Qwen3-30B-A3B-Instruct-2507-AWQ-4bit \
    --base-url http://localhost:8123 \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 50
```

---

#### 4. 查看详细 KV Cache 配置的方法

##### 方法一：启动时添加详细日志

修改你的 docker-compose.yaml，添加日志参数：

```yaml
command:
  # ... 其他参数 ...
  - "--log-level"
  - "debug"
  # 或者使用
  # - "-v"  # verbose 模式
```

##### 方法二：查看 /metrics 端点

```bash
# 获取所有指标
curl http://localhost:8123/metrics | grep -E "(cache|block|memory)"
```

关键指标解释：

|指标|含义|
|---|---|
|`vllm:num_gpu_blocks_total`|GPU 上 KV Cache 总块数|
|`vllm:num_cpu_blocks_total`|CPU 上 KV Cache 总块数|
|`vllm:gpu_cache_usage_perc`|GPU Cache 使用率|
|`vllm:prefix_cache_hit_rate`|Prefix Cache 命中率|
|`vllm:num_preemption_total`|抢占次数（KV Cache 不足时发生）|

##### 方法三：使用 vLLM 内部 API（需要修改启动方式）

如果你想获取更详细的配置，可以用 Python 直接加载模型查看：

```python
"""
离线查看模型配置（不启动服务）
"""
from vllm import LLM
from vllm.config import CacheConfig

# 只初始化，不加载权重（快速查看配置）
llm = LLM(
    model="/models/qwen3-awq",  # 本地路径
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    quantization="awq",
    # 仅用于查看配置，实际推理时去掉
    enforce_eager=True,
)

# 查看配置
print("=== Model Config ===")
print(f"Hidden size: {llm.llm_engine.model_config.hf_config.hidden_size}")
print(f"Num layers: {llm.llm_engine.model_config.hf_config.num_hidden_layers}")
print(f"Num KV heads: {llm.llm_engine.model_config.hf_config.num_key_value_heads}")
print(f"Head dim: {llm.llm_engine.model_config.hf_config.hidden_size // llm.llm_engine.model_config.hf_config.num_attention_heads}")

print("\n=== Cache Config ===")
cache_config = llm.llm_engine.cache_config
print(f"Block size: {cache_config.block_size}")
print(f"Num GPU blocks: {cache_config.num_gpu_blocks}")
print(f"Num CPU blocks: {cache_config.num_cpu_blocks}")
print(f"Cache dtype: {cache_config.cache_dtype}")

print("\n=== Scheduler Config ===")
scheduler_config = llm.llm_engine.scheduler_config
print(f"Max num seqs: {scheduler_config.max_num_seqs}")
print(f"Max num batched tokens: {scheduler_config.max_num_batched_tokens}")

# 计算 KV Cache 大小
num_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
num_kv_heads = llm.llm_engine.model_config.hf_config.num_key_value_heads
head_dim = llm.llm_engine.model_config.hf_config.hidden_size // llm.llm_engine.model_config.hf_config.num_attention_heads
dtype_bytes = 2  # FP16/BF16 = 2 bytes

kv_cache_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
print(f"\n=== KV Cache 计算 ===")
print(f"KV Cache per token: {kv_cache_per_token / 1024:.2f} KB")
print(f"KV Cache for 8192 tokens: {kv_cache_per_token * 8192 / 1024 / 1024 / 1024:.2f} GB")
```

---

#### 5. 完整 Benchmark Docker Compose（独立测试容器）

如果你想有一个专门用于 benchmark 的配置：

```yaml
# docker-compose.bench.yaml
services:
  vllm-bench:
    image: vllm/vllm-openai:v0.12.0
    container_name: vllm-bench
    volumes:
      - "D:/dev_software/AI_models/huggingface/Qwen3-30B-A3B-AWQ-4bit:/models/qwen3-awq:ro"
    gpus: all
    ipc: host
    shm_size: "16gb"
    entrypoint: ["bash"]
    stdin_open: true
    tty: true
```

使用：

```bash
# 启动 benchmark 容器
docker compose -f docker-compose.bench.yaml up -d

# 进入容器
docker exec -it vllm-bench bash

# 在容器内运行 benchmark
vllm bench throughput \
    --model /models/qwen3-awq \
    --input-len 512 \
    --output-len 256 \
    --num-prompts 50 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90

# 退出后清理
docker compose -f docker-compose.bench.yaml down
```

---

#### 总结

| 需求          | 推荐方法                                 |
| ----------- | ------------------------------------ |
| 快速查看运行参数    | `docker logs`+`/metrics`端点        |
| KV Cache 监控 | `/metrics`端点 + Python 脚本            |
| 吞吐量测试       | `vllm bench throughput`（Docker 内或本地） |
| 延迟测试        | `vllm bench latency`或 Python 脚本     |
| 详细配置查看      | Python 直接加载 LLM 对象                   |


## vllm 华为显卡部署

首先要[确认操作系统和硬件的兼容性](https://www.hiascend.com/hardware/compatibility)。（这个链接的兼容性好像更严格）

[安装华为显卡相关](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition)的：
- [显卡驱动 & 固件](https://www.hiascend.com/hardware/firmware-drivers/community)
- CANN-toolkit
- CANN-kernals
- pytorch(CANN兼容版本)。

再[安装vllm相关](https://docs.vllm.ai/projects/ascend/zh-cn/latest/installation.html#)。一定要注意[版本兼容](https://docs.vllm.ai/projects/ascend/zh-cn/latest/community/versioning_policy.html)问题。比如下表：

| vLLM Ascend | vLLM    | Python          | Stable CANN | PyTorch/torch_npu   |
| ----------- | ------- | --------------- | ----------- | ------------------- |
| v0.13.0rc1  | v0.13.0 | >= 3.10, < 3.12 | 8.3.RC2     | 2.8.0 / 2.8.0       |
| v0.11.0     | v0.11.0 | >= 3.9 , < 3.12 | 8.3.RC2     | 2.7.1 / 2.7.1.post1 |
| v0.12.0rc1  | v0.12.0 | >= 3.10, < 3.12 | 8.3.RC2     | 2.8.0 / 2.8.0       |
| v0.11.0rc3  | v0.11.0 | >= 3.9, < 3.12  | 8.3.RC2     | 2.7.1 / 2.7.1.post1 |
| v0.11.0rc2  | v0.11.0 | >= 3.9, < 3.12  | 8.3.RC2     | 2.7.1 / 2.7.1       |
| v0.11.0rc1  | v0.11.0 | >= 3.9, < 3.12  | 8.3.RC1     | 2.7.1 / 2.7.1       |

所以我如果下载8.3.RC1（华为官网推荐这个版本），那就要安装pytorch 2.7.1、vLLM v0.11.0 和 vllm-ascend v0.11.0rc1。注意如果使用docker，那么CANN是容器中的，宿主机只是显卡驱动和显卡固件要兼容，

### 准备

- 教程(选择-软件安装-安装指南): https://www.hiascend.com/document/detail/zh/CANNCommunityEdition
- 系统: 统信UOS（CentOS（RPM）体系）
- server: 华为 泰山服务器
- CPU: 鲲鹏920
- GPU: Atlas 300I Duo
- ssh信息: root@192.168.50.117 -p 36406

#### 下载文件

驱动、固件、CANN-toolkit、CANN-kernels

- Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run
- Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run

- Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run
- Ascend-cann-kernels-310p_8.3.RC1_linux-aarch64.run

#### 下载链接

- [ 驱动 & 固件 下载](https://www.hiascend.com/hardware/firmware-drivers/community)
- [CANN 下载](https://www.hiascend.com/zh/developer/download/community/result?module=cann)（最好是 [pytorch 和 CANN](https://www.hiascend.com/developer/download/community/result?module=pt+cann&pt=7.2.0&cann=8.3.RC1&product=2&model=17)都下载安装了, pytorch需要基于CANN, 后者类似CUDA）
- [cann-driver仓库(可以参考实现,但不刚需)](https://gitcode.com/cann/driver)

### 安装驱动 & 固件



1. 拷贝

```bash
scp -P 36406 Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run root@192.168.50.117:/home/LLM_project/CANN/
scp -P 36406 Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run root@192.168.50.117:/home/LLM_project/CANN/
```

2. 安装

```bash
cd /home/LLM_project/CANN
id HwHiAiUser

# 如果没有该用户
groupadd HwHiAiUser
useradd -g HwHiAiUser -d /home/HwHiAiUser -m -s /bin/bash HwHiAiUser

chmod +x Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run
chmod +x Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run

./Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run --check
./Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run --check

# 如果check ok
./Ascend-hdk-310p-npu-driver_25.3.rc1_linux-aarch64.run --full --install-for-all
./Ascend-hdk-310p-npu-firmware_7.8.0.2.212.run --full

# 重启
reboot

# 检查是否有驱动
npu-smi info
```

### 安装 CANN


1. 拷贝

```bash
scp -P 36406 Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run root@192.168.50.117:/home/LLM_project/CANN/
scp -P 36406 Ascend-cann-kernels-310p_8.3.RC1_linux-aarch64.run root@192.168.50.117:/home/LLM_project/CANN/
```

2. 安装

```bash
cd /home/LLM_project/CANN
chmod +x Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run
./Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --check

# 如果 check All good. （如果是升级，就 --upgrade）
./Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --install

# 安装完成后，若显示后文信息，则说明软件安装成功：xxx install success
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装 cann-kernels 
chmod +x Ascend-cann-kernels-<chip_type>_<version>_linux.run
./Ascend-cann-kernels-310p_8.3.RC1_linux-aarch64.run --check

# 如果 check All good. （kernels升级也要用install upgrade有问题）
./Ascend-cann-kernels-310p_8.3.RC1_linux-aarch64.run --install

# 刚才设置的环境变量需要在 bashrc/zshrc 中生效
# echo '. /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
# source ~/.bashrc
echo '. /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.zshrc
source ~/.zshrc

# 查看 CANN 版本信息
cat /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info
```


3. 安装NNAL神经网络加速库（可选）

NNAL神经网络加速库中提供了ATB（Ascend Transformer Boost）加速库和SiP（AscendSiPBoost）信号处理加速库。

加速库安装之前，需已安装同一版本的Toolkit并配置环境变量。

1. 增加对软件包的可执行权限。
```bash
chmod +x Ascend-cann-nnal_<version>_linux-aarch64.run
```

2. 安装软件包（安装命令支持`--install-path=<path>`等参数，具体使用方式请参见[参数说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0043.html)）。
```bash
./Ascend-cann-nnal_8.5.0.alpha002_linux-aarch64.run --install
```
如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。root用户：“/usr/local/Ascend”，非root用户：`_${HOME}_/Ascend`，`_${HOME}_`为当前用户目录。

3. 配置环境变量，当前以root用户安装后的默认路径为例，请用户根据加速库对应的set_env.sh的实际路径进行替换。需注意，不支持同时配置ATB和SiP的环境变量脚本。

- ATB加速库：
```bash
source /usr/local/Ascend/nnal/atb/set_env.sh

# 如果不报错
echo '. /usr/local/Ascend/nnal/atb/set_env.sh' >> ~/.zshrc
source ~/.zshrc
```
 
- SiP加速库：
```bash
source /usr/local/Ascend/nnal/asdsip/set_env.sh

# 如果不报错
echo '. /usr/local/Ascend/nnal/asdsip/set_env.sh' >> ~/.zshrc
source ~/.zshrc

# 如果报错
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
```

上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

### 部署 vllm-ascend

python 版本和docker版本是相互独立的，但都依赖上述安装的系统级软件包。docker其实就是在容器里安装好了python，所以使用docker是更方便的。

#### docker 部署

- [注意看其中的docker部分](https://docs.vllm.ai/projects/ascend/zh-cn/latest/installation.html#)
- [vllm-ascend-docker](https://quay.io/repository/ascend/vllm-ascend?tab=tags)
- [cann-py-docker](https://hub.docker.com/r/ascendai/cann/tags)
- [ascend-docker国内快速](https://quay.io/organization/ascend)

```bash
docker pull quay.io/ascend/vllm-ascend:v0.11.0rc1-310p
```

检查docker内部信息：

```bash
IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc1-310p

docker run --rm ${IMAGE} bash -lc '
set -e
ARCH=$(uname -m)
FILE=/usr/local/Ascend/ascend-toolkit/latest/${ARCH}-linux/ascend_toolkit_install.info
if [ -f "$FILE" ]; then
  echo "==> $FILE"
  grep -E "^(package_name|version|innerversion|path)=" "$FILE"
else
  echo "not found: $FILE"
  echo "try list:"
  ls -la /usr/local/Ascend/ascend-toolkit/latest/ || true
  find /usr/local/Ascend/ascend-toolkit/latest -maxdepth 2 -name ascend_toolkit_install.info -print -exec sh -c "echo ==== {}; grep -E \"^(package_name|version|innerversion|path)=\" {}" \; || true
fi
'
```

package_name=Ascend-cann-toolkit
version=8.3.RC1
innerversion=V100R001C23SPC001B235
path=/usr/local/Ascend/ascend-toolkit/8.3.RC1/aarch64-linux

#### python 部署(与docker部署替代关系)

除了下文这种方式，还可以自己根据官方支持的docker镜像来安装自己的python镜像。
- [cann-py-docker](https://hub.docker.com/r/ascendai/cann/tags)
- [ascend-docker国内快速](https://quay.io/organization/ascend)

```bash
docker pull quay.io/ascend/cann:8.3.rc1-310p-ubuntu22.04-py3.11

```
##### 安装 pytorch


1. 先安装python


版本不能随便安装，要结合所有库的支持版本的交集，比如vllm 0.11.0 要求python版本不低于3.9，那么python 3.8就不行。各种都要搜集。我综合各种条件决定安装 python 3.11.13。

```bash
# 查看 python 支持的版本
yum list | grep python

# 已当前系统支持的python39为例
sudo yum module enable -y python39
sudo yum install -y python39 python39-pip python39-setuptools python39-wheel

# 安装uv
pip3.9 install uv

# 安装虚拟环境
mkdir -p ~/.python-user
uv venv --python /usr/bin/python3.9 ~/.python-user/default

# 虚拟环境临时生效
source ~/.python-user/default/bin/activate
```

上述版本如果不能满足要求，可以选择源码安装（但是gcc 版本不太行，安装gcc高版本要见《llama.cpp 本地编译(华为)》）
```bash
# 安装 python 编译依赖
sudo yum install -y \
  gcc-toolset-12-gcc \
  gcc-toolset-12-gcc-c++ \
  gcc-toolset-12-libstdc++-devel

sudo yum install -y make wget \
  zlib-devel bzip2-devel xz-devel \
  readline-devel sqlite-devel \
  openssl-devel libffi-devel \
  ncurses-devel tk-devel gdbm-devel
```

2. 再安装 pytorch

- [华为官网-安装pytorch教程](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)
	再次强调，注意版本！

3. 部署 pip 版 vllm

- [注意看其中的pip部分](https://docs.vllm.ai/projects/ascend/zh-cn/latest/installation.html#)

### vllm 华为 bug

- [LinearOperation CreateOperation failed](https://github.com/Ascend/pytorch/issues/94)
- [vllm-ascend:v0.11.0rc2 qwen3-next-80B OOM](https://github.com/vllm-project/vllm-ascend/issues/4474)



## llama.cpp 华为显卡部署

同样是有两种方式，推荐是docker，因为本地gcc相关的环境容易改出问题

###  1-1 llama.cpp docker (华为)

- [llama.cpp-docker](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)

```bash
git clone https://git.ustc.edu.cn/ustc-os-lab/llama.cpp.git
cd llama.cpp

# 编辑这个dockerfile 修改基础镜像的版本适配本机安装的CANN, 然后 :wq 保存
vim .devops/llama.cpp.cann.Dockerfile

docker build -f .devops/cann.Dockerfile -t llama-cpp-cann:full --target full .
```

###  1-2 llama.cpp 本地编译(华为)

- [llama.cpp-CANN](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)

```bash
# clone 代码 (国内镜像)
git clone https://git.ustc.edu.cn/ustc-os-lab/llama.cpp.git
cd llama.cpp

# 查看可用版本
yum list | grep gcc-toolset

# 安装
sudo yum install -y \
  gcc-toolset-12-gcc \
  gcc-toolset-12-gcc-c++ \
  gcc-toolset-12-libstdc++-devel

# gcc-toolset-12 (加入环境变量, 如果想要持久生效加入 .zshrc)  
export GCC12_ROOT=/opt/UOS/gcc-toolset-12/root/usr
export PATH="$GCC12_ROOT/bin:$PATH"

# 库环境变量可能会有问题: 某些程序本来应该加载系统的库版本，但因为把 toolset 的 libstdc++ 放前面，程序加载了不同版本的 libstdc++.so.6，少数情况下会出现兼容性问题
export LD_LIBRARY_PATH="$GCC12_ROOT/lib64:${LD_LIBRARY_PATH}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh --force

cmake -B build \
-DGGML_CANN=ON \
-DCMAKE_BUILD_TYPE=Release \
-DSOC_TYPE=ascend${CHIP_TYPE} \

cmake --build build --config Release -j$(nproc)

# 运行可能还有问题, 因为有些依赖的动态库可能版本不兼容

```

### 2. llama.cpp 模型

- [unsloth-qwen3](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)

- 它的入口通常是 llama-cli

- CANN 后端对量化格式有限制
    仓库的 CANN 指南明确写了：目前只支持 **FP16 / Q4_0 / Q8_0**。
    这点很关键：Hugging Face 上很多 GGUF 是 Q4_K_M、Q5_K_M、IQ* 等，你在 Ascend/CANN 上不一定能用（至少这份指南口径是不支持）。

- lama.cpp 通用要求：模型必须是 GGUF
    llama.cpp 官方 README：llama.cpp 需要模型是 **GGUF**；其他格式要用仓库里的 convert_*.py 脚本转换。

```bash
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /home/LLM_project/models/GUFF/Qwen3-30B-A3B-Q4_0

hf download unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_0.gguf \
   --local-dir /home/LLM_project/models/GUFF/Qwen3-30B-A3B-Q4_0
```


### 3-1 docker 部署

- 《AI-config/llama-cpp-docker/llama-cpp-ascend.yml》

```bash
docker compose -f llama-cpp-ascend.yml up -d
```

### 3-2 本地运行

```bash
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

### 4 llama.cpp MoE bug


```
load_tensors:        CANN0 model buffer size =    48.80 MiB   ← NPU 只放了 48MB
load_tensors:  CPU_AARCH64 model buffer size = 15390.00 MiB  ← 模型主体还在 CPU
```

虽然显示 `offloaded 49/49 layers`，但实际上 **MoE 的专家层没有真正卸载到 NPU**。

这是 llama.cpp CANN 后端对 **MoE 架构支持不完整** 的问题。Qwen3-30B-A3B 有 128 个专家，这些专家权重仍在 CPU 上。

#### 验证方式

```
graph splits = 483
```

483 次 CPU/NPU 切换，说明大量计算在 CPU 和 NPU 之间来回跳。

#### 解决方案

**方案 1：换非 MoE 模型**（推荐）

在 310P3 上跑 **Qwen2.5-14B** 或 **Qwen2.5-7B** 这类 Dense 模型会快很多：

```bash
# 下载 Qwen2.5-14B-Instruct Q4_K_M（约 8.5GB，完全放得下）
```

**方案 2：继续用 vLLM**

你之前用的 vLLM 对华为 NPU 和 MoE 架构支持应该更成熟。如果一定要跑 Qwen3-30B-A3B，vLLM 可能是更好的选择。

**方案 3：等 llama.cpp 更新**

CANN 后端对 MoE 的支持还在完善中，可以关注 [llama.cpp CANN 相关 issue](https://github.com/ggerganov/llama.cpp/issues)。

---



## tensorflow 华为显卡

- [ascend-tensorflow仓库](https://gitee.com/ascend/tensorflow)


## 模型与显卡的性能指标

### 模型

模型本身的参数：
- 模型的参数量、量化bit位数、上下文、激活的参数量。

模型使用的参数：
- 单访问 tokens/s
- 多用户并发总 tokens/s
- first token 延迟
- 100token输入 & 500token 输出单用户耗时

### 显卡

单卡的性能（q4 q8 的tops）、单卡的显存容量、单卡的显存带宽

多卡的通信瓶颈、有无nvlink的差距；nvlink的支持情况。


## llm部署优化


原则上应该是尽量单卡能运行模型，然后并发多卡独立部署再用ngi 做负载均衡


单卡够用时的选择，不需要 tensor-parallel 的情况；如果单卡显存足够，`--tensor-parallel-size 2` 反而可能**降低性能**，因为：

- 两卡之间有通信开销（NVLink/PCIe）
- 同步等待会增加延迟

想用两张卡提高并发，正确做法是**启动两个独立实例**，每个实例用一张卡：

```bash
# 实例1：使用 GPU 0，端口 8000
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --port 8000

# 实例2：使用 GPU 1，端口 8001
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --port 8001
```

然后用 Nginx 做负载均衡：
```nginx
upstream vllm_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

server {
    listen 80;
    location / {
        proxy_pass http://vllm_backend;
    }
}
```

|场景|推荐方案|原因|
|---|---|---|
|单卡放不下模型|`--tensor-parallel-size 2`|必须拆分|
|单卡够用，想提高并发|两个独立实例 + 负载均衡|吞吐量更高，无通信开销|
|单卡够用，想降低单次延迟|`--tensor-parallel-size 2`|可能略有提升，但不明显|

### vLLM 配置侧的关键优化点


0) 先把“版本与已知坑”固定住

- Qwen3-Next 官方明确要求 vllm>=0.10.2。 
- vLLM 的 Qwen3-Next recipes 里提到：如果遇到 CUDA illegal memory access，可加 --compilation_config.cudagraph_mode=PIECEWISE。 
- vLLM 的核心吞吐来自连续批处理、PagedAttention、chunked prefill 等能力。 


1) 你这种“每次几千 token”的负载，第一优先级是避免 preemption（抢占/重算）

抢占会导致重算，直接把尾延迟打爆。vLLM 文档建议的处理手段很明确：


- 提高 gpu_memory_utilization 给 KV cache 更多显存
- 或降低 max_num_seqs / max_num_batched_tokens 减少同批并发占用的 KV 空间 


实操建议（思路，不是唯一答案）

  - 单实例独占 GPU：gpu_memory_utilization 往 0.92~0.97 区间试（视你显存余量而定）。 
- 只要监控里出现频繁 preempt/recompute，就不要继续“加并发”，而是先把 KV 空间扩出来（或把上下文/批次控下来）。


2) 开启并调好 chunked prefill（你的场景几乎必开）


RAG 长 prompt 的典型问题是：长 prefill 会压住 decode，短请求被“堵车”。chunked prefill 的目的就是把长 prefill 切块，和 decode 交错调度，从而同时改善吞吐与延迟。


你要关注的不只是 --enable-chunked-prefill，还包括“长 prompt 并发 prefill 的上限”，避免一堆超长 prompt 同时进来把 GPU 步长全占了：

- --max-num-partial-prefills
- --max-long-partial-prefills
- --long-prefill-token-threshold（多版本文档/参数解释一致） 

经验法则：

- RAG 高并发时，把 “long partial prefills” 设得比 “partial prefills” 更保守，让短请求更容易插队，p95 会明显好看。 

3) 打开 Automatic Prefix Caching，并让你的请求真正“吃到缓存”

vLLM 的 APC（自动前缀缓存）会缓存已处理前缀对应的 KV blocks，新请求如果共享同一前缀即可跳过那段 prefill，属于典型“几乎白给”的优化，且不改变输出。

但注意：APC 的收益取决于前缀共享率。想吃到缓存，你需要在 RAG 应用侧配合（后面会讲）。

另外，如果你多实例扩容，要尽量让同前缀请求路由到同一实例，否则缓存命中率会被打散。Ray Serve 就提供了面向 prefix caching 的路由策略思路（强调“缓存命中比完美负载均衡更重要”）。



4) 用 

max_num_batched_tokens

 / 

max_num_seqs

 控吞吐-延迟的杠杆


这俩是 vLLM 服务端最核心的“批量化闸门”：


- max_num_seqs 控同一轮调度里最多并发序列数
- max_num_batched_tokens 控每轮最多处理的总 token 预算
    文档与参数列表里明确给出了这两个概念。 

建议调参顺序（适用于你的“长输入+多用户”）：

1. 先把 max_model_len（上下文上限）设到你业务真正需要的值，别为了“以防万一”拉满；这会直接决定 KV 预算天花板。
2. 开 chunked prefill 后，先用偏保守的 max_num_seqs 保 p95，再逐步加 max_num_batched_tokens 拉吞吐。
3. 一旦出现 preemption 或 GPU KV cache usage 接近 1 且排队加长，就回退并发或缩短上下文。

4) 监控一定要上：用 /metrics 盯住“是不是在浪费算力”


vLLM 的指标文档把指标分为 server-level 与 request-level 两类，非常适合用来定位是“KV 不够”“长 prompt 堵车”“缓存没命中”还是“调度参数太保守”。

并且 vLLM 的 OpenAI server 会暴露 Prometheus 格式指标，官方也有 Prometheus+Grafana 的示例。


你这种场景建议重点盯：

- vllm:gpu_cache_usage_perc（KV 使用率）与 vllm:gpu_prefix_cache_hit_rate（前缀缓存命中） 
- TTFT、TPOT（每 token 延迟）相关的 request-level 直方图
- preemption/recompute 相关计数（如果有，就说明 KV/并发配置不合理） 

6) 两张 GPU 时可考虑“Prefill/Decode 解耦”，但要当成高级选项


如果你经常有“极长 prompt + 同时很多短请求”，单引擎即便 chunked prefill 也可能出现明显干扰。vLLM 提供了把 prefill 和 decode 放到不同实例/GPU 的实验性方案（KV 在两者间传输）。

这种思路也被社区文章用来解释“长 prompt 阻塞短请求”，并给出拆分部署的方向。

但它是“工程换收益”：会引入 KV 传输与更多运维复杂度，且文档明确是 experimental。

建议你先把：APC + chunked prefill + 并发闸门 + RAG 侧减 prompt/减调用 做到位，再评估是否需要上解耦。
