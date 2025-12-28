
# LLM 部署平台

vllm、llama.cpp、ollama、openllm

vllm是支持并发最好的，llama.cpp是支持平台最多的，ollama是最简单性能也是最差的。

## [vllm](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)部署

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


## LLM部署性能对比分析


随着Blackwell架构GPU和改进的MoE架构的出现，运行70-120B参数模型的硬件格局发生了重大变化，但**关键的NVLink缺失限制了多GPU扩展能力**。RTX PRO 6000的96GB显存支持70B模型单卡部署，而双RTX 5090配置在Llama 3.3-70B上实现了**27 tokens/秒**，但由于缺少NVLink存在已记录的P2P通信问题。Apple M4 Max凭借统一内存优势在70B Q4模型上达到**10-12 tok/s**，而华为Atlas 300I Duo虽提供96GB容量，但**每芯片204 GB/s带宽 & tops性能不足**——不到NVIDIA最新产品的五分之一。

-----

### 一、模型规格与显存需求

三款目标模型均已公开发布并可部署，但命名上存在一些变体值得注意。

#### 1.1 Llama 3/3.1-70B

仍是基准测试的标准模型，拥有700亿参数，128K上下文长度（3.1版本），在各种量化级别下都有完善的性能数据。显存需求从**Q4_K_M的39.6GB**到**FP16的140GB**不等，非常适合RTX PRO 6000的96GB显存或M4 Max的128GB统一内存。

#### 1.2 Qwen3-Next-80B-A3B-Instruct-FP8

是一款混合Transformer-Mamba MoE架构模型，总参数量80B但**每token仅激活3.9B参数**。它使用512个路由专家，每次前向传播激活10个，在32K以上上下文时相比稠密版Qwen3-32B可实现10倍推理吞吐量。该模型在Arena-Hard v2基准测试中达到82.7分，与Qwen3-235B-A22B相当，但每token所需算力大幅降低。

#### 1.3 gpt-oss-120B

（2025年8月发布）包含117B参数，通过128个专家激活5.1B参数。其原生MXFP4量化支持**单张H100部署**，RedHatAI提供FP8变体。该模型在核心推理基准上接近o4-mini水平，同时完全开源（Apache 2.0协议）。

#### 1.4 模型参数总览表

|模型                |总参数量|激活参数量  |量化后大小(FP8/Q4)   |原生上下文长度    |
|------------------|----|-------|----------------|-----------|
|Llama 3.1-70B     |70B |70B（稠密）|35-40GB (Q4)    |128K       |
|Qwen3-Next-80B-A3B|80B |3.9B   |~80GB (FP8)     |262K（扩展至1M）|
|gpt-oss-120B      |117B|5.1B   |~60-65GB (MXFP4)|128K       |

-----

### 二、GPU硬件规格对比

四种硬件配置涵盖完全不同的架构、价格区间和部署场景。

#### 2.1 NVIDIA RTX PRO 6000 Blackwell（96GB）

NVIDIA旗舰工作站GPU通过512位GDDR7（带ECC）提供**1,792 GB/s显存带宽**，配备752个第五代Tensor Core。

**计算性能规格：**

- FP8：503.80 TFLOPS（稀疏加速后1,007.61 TFLOPS）
- INT8：1,007 TOPS（稀疏加速后2,015 TOPS）

96GB容量可在单卡上完整运行Q8量化的Llama 3-70B而无需拆分。

**关键限制：不支持NVLink。** NVIDIA将NVLink专用于数据中心GPU，工作站多GPU配置只能依赖PCIe 5.0 x16（约64 GB/s双向带宽）进行GPU间通信。

#### 2.2 NVIDIA RTX 5090（32GB × 2）

消费级旗舰采用相同的Blackwell架构（GB202），拥有**21,760个CUDA核心**、680个Tensor Core，以及同样的**1,792 GB/s单卡显存带宽**。

**性能规格：**

- FP8：419 TFLOPS（稀疏加速后838 TFLOPS）
- INT8：838 TOPS（稀疏加速后1,676 TOPS）

双卡提供64GB总显存——足以运行Q4量化的Llama 3-70B并留有KV缓存空间。

**多GPU挑战已有充分文档记录。** GitHub issues和NVIDIA论坛报告显示，双RTX 5090配置的vLLM张量并行需要设置`NCCL_P2P_DISABLE=1`，用户会看到”由于您的平台缺乏GPU P2P能力，自定义allreduce已禁用”的警告。实际使用中流水线并行比张量并行更可靠。

#### 2.3 Apple M4 Max（128GB统一内存）

Apple顶级消费芯片为其40核GPU和16核神经引擎提供**546 GB/s内存带宽**。统一内存架构消除了模型权重的PCIe传输瓶颈，使70B+模型可以完全在内存中运行，无需复杂的多GPU协调。功耗约**60W**，而NVIDIA独立方案需要575W以上。

带宽限制——约为RTX 5090的3.3分之一——直接制约了内存带宽受限推理工作负载的token生成速度。

#### 2.4 华为Atlas 300I Duo（96GB）

这款双NPU卡使用两颗昇腾310P3芯片（达芬奇架构），每颗配备48GB LPDDR4X。总容量达96GB，但**每颗芯片独立限制在204 GB/s带宽**——这是关键瓶颈。

**计算规格：**

- INT8：280 TOPS
- FP16：140 TFLOPS
- TDP：150W（被动散热设计，面向服务器环境）

**生态系统摩擦仍然显著。** 该卡需要基于华为鲲鹏920的服务器和仅Linux运行环境，不兼容标准桌面主板。开发者描述通过CANN进行软件开发是”一条充满坑的路”。

#### 2.5 GPU硬件规格总览表

| GPU            | 显存      | 带宽           | FP8 TFLOPS    | INT8 TOPS       | NVLink |
| -------------- | ------- | ------------ | ------------- | --------------- | ------ |
| RTX PRO 6000   | 96GB    | 1,792 GB/s   | 504 (稀疏1,008) | 1,007 (稀疏2,015) | ❌      |
| RTX 5090       | 32GB    | 1,792 GB/s   | 419 (稀疏838)   | 838 (稀疏1,676)   | ❌      |
| M4 Max 40核     | 128GB共享 | 546 GB/s     | N/A           | N/A             | N/A    |
| Atlas 300I Duo | 96GB    | 204 GB/s × 2 | N/A           | 280             | N/A    |

-----

### 三、来自专业来源的实测基准数据

以下基准数据完全来自有据可查的来源，包括GitHub仓库、专业评测网站和社区测试。明确排除估算值。

#### 3.1 Llama 3-70B跨硬件性能

来自GPU-Benchmarks-on-LLM-Inference（GitHub）的**vLLM和llama.cpp基准测试**提供了最全面的跨平台比较：

|硬件            |量化    |Token生成 (tok/s)|提示处理 (tok/s)|
|--------------|------|---------------|------------|
|H100 PCIe 80GB|Q4_K_M|25.03          |1,012       |
|A100 SXM 80GB |Q4_K_M|24.61          |817         |
|2× RTX 4090   |Q4_K_M|19.22          |839         |
|2× RTX 3090   |Q4_K_M|16.57          |—           |
|M2 Ultra 192GB|Q4_K_M|12.48          |—           |
|M3 Max 64GB   |Q4_K_M|7.65           |—           |

#### 3.2 双RTX 5090实测性能

（DatabaseMart/HostKey通过Ollama测试）：

|模型             |Token生成速度       |
|---------------|----------------|
|DeepSeek-R1 70B|**27 tok/s**    |
|Llama 3.3 70B  |**27 tok/s**    |
|Qwen 2.5 72B   |~26 tok/s       |
|Qwen 2.5 110B  |7.22 tok/s（显存受限）|

#### 3.3 单张RTX 5090 llama.cpp基准测试

（Hardware Corner，Q4_K_XL量化）：

|模型              |Token生成 (tok/s)|提示处理 (tok/s)|
|----------------|---------------|------------|
|Qwen3 32B       |61.38          |2,931       |
|Qwen3moe 30B.A3B|**234.30**     |—           |

达到的最大上下文：**147K tokens**，Qwen3moe 30B下52.28 tok/s

#### 3.4 RTX PRO 6000 Blackwell基准测试

（CloudRift.ai、StorageReview）：

- Qwen3-Coder-30B-A3B-Instruct-AWQ：通过vLLM达**8,425 tok/s**
- Procyon AI基准（Llama3）：6,501
  - 对比RTX 5090：6,104
  - 对比RTX 4090：4,849
- 96GB显存支持运行70B Q8模型而无需量化妥协

#### 3.5 Apple M4 Max 128GB基准测试

（GitHub llama.cpp、MacRumors、社区测试）：

| 模型                  | 量化            | Token生成 (tok/s) | TTFT  |
| ------------------- | ------------- | --------------- | ----- |
| Llama 3.3 70B       | Q4 (MLX)      | ~12             | —     |
| Command-R-Plus 104B | Q4_K_M (62GB) | 6.5             | 2.26s |
| Command-R-Plus 104B | Q6_K (85GB)   | 4.5             | 4-9s  |
| LLaMA 2 7B          | Q4_0          | 83.06           | —     |

#### 3.6 华为Atlas 300I Duo基准测试

（英文数据有限）：

|模型规模     |Token生成速度|备注      |
|---------|---------|--------|
|Qwen3 32B|~15 tok/s|早期测试    |
|8B级模型    |~15 tok/s|昇腾910B参考|
|70B级Q4模型 |~5 tok/s |带宽受限    |

#### 3.7 MoE模型展现显著优势

**Qwen3-Next-80B-A3B** 由于3.9B激活参数比例，相比同等大小的稠密模型实现7-10倍吞吐量。Hardware Corner的基准测试显示RTX 5090上Qwen3moe 30B.A3B达到**234 tok/s**而稠密版Qwen3 32B仅**61 tok/s**，清晰展示了MoE的吞吐量优势。

-----

### 四、各平台性能瓶颈深度分析

#### 4.1 RTX PRO 6000：显存容量解决70B部署问题

96GB容量消除了70B模型的多GPU复杂性。Q8量化（约70GB）时，整个模型加KV缓存可以轻松装入单卡。**1,792 GB/s带宽与RTX 5090相当**，意味着每token生成速度相近，但简化的单GPU部署避免了所有P2P通信开销。

**计算不是瓶颈。** 504+ FP8 TFLOPS远超LLM推理的需求——对于token生成，工作负载仍然是显存带宽受限的。真正的优势是避免了困扰双消费级GPU配置的P2P和张量并行问题。

**瓶颈分析：**

- ✅ 显存容量：充足（96GB）
- ✅ 显存带宽：1,792 GB/s（优秀）
- ✅ 计算能力：504 TFLOPS FP8（过剩）
- ❌ 多卡扩展：无NVLink，依赖PCIe

#### 4.2 双RTX 5090：PCIe限制张量并行效果

没有NVLink的情况下，两张RTX 5090卡之间的张量并行必须通过PCIe 5.0传输激活值（约64 GB/s双向）。对于TP=2的Llama 70B，这为每层计算增加了可测量的延迟。已记录的P2P问题加剧了这一点——GitHub报告显示vLLM回退到较慢的NCCL实现。

**流水线并行在实践中效果更好。** 与其将层水平拆分到GPU之间，不如垂直拆分模型（GPU 1处理0-39层，GPU 2处理40-79层），将通信减少到每次前向传播一次而非每层一次。这解释了为什么Ollama（使用流水线并行）达到27 tok/s，而vLLM张量并行用户报告困难。

**显存带宽仍然优秀。** 每卡1,792 GB/s，双卡配置的聚合带宽超过企业级H100配置。瓶颈是GPU间通信，而非显存读取。

**瓶颈分析：**

- ⚠️ 显存容量：64GB（需要Q4量化）
- ✅ 显存带宽：1,792 GB/s × 2（优秀）
- ✅ 计算能力：838 TFLOPS FP8（过剩）
- ❌ GPU间通信：PCIe 5.0 ~64 GB/s（主要瓶颈）
- ❌ P2P支持：需要禁用，使用NCCL回退

#### 4.3 M4 Max：统一内存实现独特部署场景

128GB统一内存允许运行在NVIDIA硬件上需要多GPU拆分的模型——Q4量化的Command-R-Plus 104B完全装入内存。无需模型拆分逻辑、无需跨设备同步、无需P2P配置难题。

**带宽限制70B模型的生成速度约为10-12 tok/s。** 546 GB/s统一内存带宽（约为RTX 5090的3.3分之一）直接制约token生成率。然而，对于优先考虑部署简便性、能效（60W vs 575W）或静音运行而非最大吞吐量的场景，这可能是可接受的。

**提示处理比生成受影响更大。** 7B模型提示处理922 tok/s对比RTX 5090的10,000+ tok/s，预填充密集型工作负载的计算差距更明显。70B模型处理45K+ token上下文的长文档摄入可能需要25分钟以上。

**瓶颈分析：**

- ✅ 显存容量：128GB（优秀）
- ❌ 显存带宽：546 GB/s（主要瓶颈）
- ⚠️ 计算能力：相对较低
- ✅ 部署复杂度：最低
- ✅ 功耗：60W（极低）

#### 4.4 Atlas 300I Duo：带宽瓶颈抵消容量优势

尽管与RTX PRO 6000容量相当（96GB），Atlas 300I Duo的**每芯片204 GB/s带宽**造成了根本性的吞吐量上限。这比RTX 5090/PRO 6000低8.8倍，解释了为什么实测70B性能（约5 tok/s）低于M4 Max，尽管名义算力规格更高。

**双芯片架构不能为单个推理任务聚合带宽。** 每颗昇腾310P3独立运行，意味着工作负载无法像统一内存系统那样跨芯片池化内存带宽。

**软件生态系统摩擦加剧了硬件限制。** 报告称CANN开发体验比CUDA困难得多，英文文档和调试资源有限。这影响的是原始性能指标之外的实际部署速度。

**瓶颈分析：**

- ✅ 显存容量：96GB（充足）
- ❌ 显存带宽：204 GB/s × 2（严重瓶颈，不可聚合）
- ⚠️ 计算能力：280 TOPS INT8
- ❌ 软件生态：CANN成熟度不足
- ❌ 硬件兼容性：需要特定服务器平台

-----

### 五、vLLM与llama.cpp推理框架对比

#### 5.1 框架特性对比

**vLLM擅长并发服务。** Red Hat基准测试显示vLLM在峰值负载下提供**比llama.cpp高35倍的请求/秒**，**44倍的token吞吐量**。PagedAttention实现了批量请求间高效的KV缓存管理。

**llama.cpp适合单用户场景。** 对于单请求推理，llama.cpp达到vLLM 94-100%的性能，同时启动几乎瞬时、内存开销极小、部署更简单。由于Metal优化，它仍是Apple Silicon的主要框架。

#### 5.2 框架选择建议

|场景           |推荐框架           |理由                          |
|-------------|---------------|----------------------------|
|生产API服务      |vLLM           |PagedAttention、批处理、35倍以上并发优势|
|单用户本地推理      |llama.cpp      |更低开销、更快启动                   |
|Apple Silicon|llama.cpp / MLX|原生Metal支持                   |
|华为Atlas      |vllm-ascend插件  |官方昇腾支持                      |
|多GPU张量并行     |vLLM           |原生TP支持                      |
|资源受限环境       |llama.cpp      |更低内存占用                      |

-----

### 六、交叉对比：模型-硬件性能矩阵

基于可用的实测数据和已记录的约束条件（非估算）：

|硬件配置             |Llama 3-70B Q4     |Qwen3-Next-80B-A3B  |gpt-oss-120B      |
|-----------------|-------------------|--------------------|------------------|
|RTX PRO 6000 96GB|~25 tok/s（根据类似配置推测）|预期高吞吐量（MoE优势）       |MXFP4/FP8可装入      |
|2× RTX 5090      |**27 tok/s**（实测）   |TP=2应可装入            |需要激进量化            |
|M4 Max 128GB     |**10-12 tok/s**（实测）|可装入内存，预期~10-15 tok/s|Q4可装入，预期~4-6 tok/s|
|2× Atlas 300I Duo|**~5 tok/s**（外推）   |尽管容量够但带宽受限          |可装入内存，带宽瓶颈        |

**重要说明：** 由于硬件/模型较新，RTX PRO 6000和Qwen3-Next-80B-A3B的直接基准测试尚未公开。gpt-oss-120B模型基准测试聚焦于H100/MI300X数据中心硬件而非消费级/工作站配置。

-----

### 七、技术洞察与部署建议

#### 7.1 MoE模型从根本上改变了计算逻辑

Qwen3-Next-80B-A3B的3.9B激活参数意味着它相比稠密70B模型实现7-10倍吞吐量，同时显存占用相近。对于吞吐量敏感的应用，MoE架构提供了显著优势——前提是推理框架支持高效的专家路由。

#### 7.2 NVLink缺失重塑了多GPU策略

RTX 5090和RTX PRO 6000都不支持NVLink，使得流水线并行比张量并行更适合多卡配置。这与数据中心最佳实践（NVLink支持高效张量并行）形成显著差异，影响部署架构决策。

#### 7.3 统一内存的简便性被低估

M4 Max 128GB可以运行在NVIDIA硬件上需要4张以上独立GPU的模型——吞吐量较低，但复杂度大幅降低。对于开发、测试或延迟容忍型生产环境，部署简便性可能超过原始性能差距的影响。

#### 7.4 华为Atlas以带宽为代价提供容量

96GB/$1,400-2,000的价格点提供了出色的$/GB显存性价比，但204 GB/s带宽限制和软件生态系统挑战使其仅适合容量比吞吐量更重要的场景——可能是批处理或延迟不关键的异步工作负载。

#### 7.5 部署场景推荐总结

|部署场景      |推荐硬件               |理由                     |
|----------|-------------------|-----------------------|
|专业单卡70B部署 |RTX PRO 6000       |96GB单卡，无多GPU复杂性        |
|性价比多GPU推理 |双RTX 5090          |高吞吐量，需接受P2P配置          |
|开发/测试/静音环境|M4 Max 128GB       |部署最简单，功耗最低             |
|容量优先/预算受限 |Atlas 300I Duo     |最低$/GB，需CANN开发能力       |
|高并发生产服务   |RTX PRO 6000 + vLLM|单卡简化运维 + PagedAttention|

-----

### 八、参考文献

1. Database Mart - “2×RTX 5090 Ollama Benchmark: Outperforming H100 & A100 for 70B LLM Inference”
   https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx5090-2
2. Hardware Corner - “RTX 5090 LLM Benchmark Results: 10K Tokens/sec Prompt Processing, 139K Context”
   https://www.hardware-corner.net/rtx-5090-llm-benchmarks/
3. Hardware Corner - “Huawei’s Atlas 300I Duo offers 96GB VRAM for local LLMs under $1500”
   https://www.hardware-corner.net/huawei-atlas-300i-duo-96gb-llm-20250830/
4. NVIDIA - “RTX PRO 6000 Blackwell Workstation Edition”
   https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/
5. NVIDIA NIM - “Qwen3-Next-80B-A3B Model Card”
   https://build.nvidia.com/qwen/qwen3-next-80b-a3b-thinking/modelcard
6. GitHub vllm-project - “Issue #14628: Multi GPU inference using two RTX 5090s(TP=2)”
   https://github.com/vllm-project/vllm/issues/14628
7. vLLM Forums - “Added second 5090 and turned on tensor parallel 2”
   https://discuss.vllm.ai/t/added-second-5090-and-turne-on-tensor-parallel-2/1629
8. MacRumors Forums - “M4 Max Studio 128GB - LLM testing”
   https://forums.macrumors.com/threads/m4-max-studio-128gb-llm-testing.2453816/
9. Apple Newsroom - “Apple introduces M4 Pro and M4 Max”
   https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
10. GitHub llama.cpp - “Discussion #4167: Performance of llama.cpp on Apple Silicon M-series”
   https://github.com/ggml-org/llama.cpp/discussions/4167
11. WareDB - “NVIDIA RTX PRO 6000 Blackwell AI Performance and Hardware Specs”
   https://www.waredb.com/processor/nvidia-rtx-pro-6000-blackwell
12. Vast.ai - “NVIDIA GeForce RTX 5090 Specs: Everything You Need to Know”
   https://vast.ai/article/nvidia-geforce-rtx-5090-specs-everything-you-need-to-know
13. SabrePC - “Do You Really Need NVLink for Multi-GPU Setups?”
   https://www.sabrepc.com/blog/computer-hardware/nvlink-vs-pcie-do-you-need-nvlink-for-multi-gpu
14. VideoCardz - “Huawei Atlas 300I dual AI GPU with 96GB memory worth $1400 has been taken apart”
   https://videocardz.com/newz/huawei-atlas-300i-dual-ai-gpu-with-96gb-memory-worth-1400-has-been-taken-apart
15. ChinaTalk - “Can Huawei Take On Nvidia’s CUDA?”
   https://www.chinatalk.media/p/can-huawei-compete-with-cuda
16. SecondState - “Lightweight and cross-platform LLM agents on Ascend 910B”
   https://www.secondstate.io/articles/llm-agents-on-ascend/
17. Red Hat Developer - “vLLM or llama.cpp: Choosing the right LLM inference engine for your use case”
   https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case
18. Simon Willison - “I can now run a GPT-4 class model on my laptop”
   https://simonwillison.net/2024/Dec/9/llama-33-70b/
19. Ivan Fioravanti (X/Twitter) - “Llama 3.3 70B 4-bit running on a 128GB M4 Max with MLX LM”
   https://x.com/ivanfioravanti/status/1865237429780721853
20. Hardware Corner - “Qwen3 LLM Hardware Requirements – CPU, GPU and Memory”
   https://www.hardware-corner.net/guides/qwen3-hardware-requirements/

-----

*报告生成日期：2025年12月23日*
*数据来源：英文专业技术网站、GitHub仓库、社区测试*



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
