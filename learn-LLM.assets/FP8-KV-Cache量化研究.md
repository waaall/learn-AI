# FP8 KV Cache 量化研究

- [llm-inference-series-4-kv-caching](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8)

**Qwen3-Next-80B-A3B 和 GPT-oss-120B 均无法使用 FP8 KV cache 量化**，原因是它们采用了新型混合注意力架构。尽管两者都有官方 FP8 权重量化版本，但 KV cache 量化仍无法使用。这对于希望通过 KV cache 量化优化显存的工程师来说是一个关键限制。对于标准注意力架构的模型（如 Llama-3.1、Qwen3-235B），FP8 KV cache 可实现 **<1% 精度损失** 和 **50% 显存节省**，但校准缩放因子（calibrated scales）对于最佳精度至关重要。

-----

## Qwen3-Next-80B-A3B：混合注意力架构的障碍

**Qwen3-Next-80B-A3B** 是阿里巴巴 2025 年 9 月发布的突破性混合架构模型：总参数 80B，但每个 token 仅激活 **3B 参数**。该模型采用 48 层的新颖布局：`12 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，原生支持 **262,144 token 上下文**，通过 YaRN 可扩展至超过 100 万 tokens。

**FP8 KV cache 量化在该模型上明确不可用**。vLLM issue #26646（2025年10月）记录了致命错误：`ValueError: type fp8e4nv not supported in this architecture`。根本原因是混合注意力机制——Gated DeltaNet（线性注意力变体）与 Gated Attention 的组合——与标准 FP8 KV cache 反量化内核不兼容。级联注意力优化对混合模型被禁用，FlashInfer 后端会产生乱码输出。

|配置          |支持状态                                           |备注                         |
|------------|-----------------------------------------------|---------------------------|
|FP8 权重量化    |✅ 官方支持 (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`)|细粒度 FP8，block size 128     |
|FP8 KV cache|❌ 不支持                                          |混合注意力不兼容                   |
|vLLM 部署     |⚠️ 需要 nightly 版本                                |仅使用 `--kv-cache-dtype auto`|

**解决方案**：使用官方 FP8 权重 checkpoint 部署（单张 H100 约需 ~76GB 显存），但 KV cache 类型设为 auto。对于 FP8 权重量化，benchmark 退化很小——Qwen 官方声明结果”来自量化前的原始 bfloat16 模型”。

-----

## GPT-oss-120B：同样受限于混合注意力

OpenAI 的 **GPT-oss-120B**（2025年8月，Apache 2.0 许可证）是一个 **117B 参数的 MoE 模型**，包含 128 个专家，每次激活 4 个，实际激活参数为 **5.1B**。在 HuggingFace 上以 `openai/gpt-oss-120b` 提供，下载量超过 381 万次，在 benchmark 上接近 o4-mini 水平：**MMLU 90.0%**、**AIME 2024 96.6%**、**GPQA Diamond 80.9%**。

**FP8 KV cache 会导致运行时崩溃**（vLLM issue #23832）。服务器可以使用 `--kv-cache-dtype fp8` 成功启动，但在推理时会失败，报错 `RuntimeError: query and key must have the same dtype`。问题根源是 Flash Attention 3 与该模型独特架构之间的内核不兼容：

- **注意力 sink 向量**：训练的 per-query-head sink 向量干扰 FP8 转换
- **混合注意力模式**：以 1:1 比例交替使用全注意力和 128-token 滑动窗口
- **自定义 KV cache 分配器**：vLLM 的新技术在全注意力/滑动窗口层之间动态共享 cache，破坏了 FP8 反量化路径

该模型原生使用 **MXFP4 量化**（4-bit fp4 e2m1，带 block FP32 scales）处理 MoE 权重，总大小保持在 ~63GB，无需 FP8 KV cache 优化即可在单张 H100 上运行。

-----

## vLLM 的 FP8 KV cache 对标准架构有效

对于采用传统注意力的模型（Llama-3.1、Mistral、Qwen3-235B），vLLM 的 FP8 KV cache 实现可带来显著收益。该框架支持来自 Open Compute Project 的两种 FP8 格式：

|格式          |指数位|尾数位|动态范围     |使用场景     |
|------------|---|---|---------|---------|
|**E4M3**（默认）|4 位|3 位|±240.0   |更高精度，推荐使用|
|**E5M2**    |5 位|2 位|与 FP16 相同|更大范围，较低精度|

**硬件加速**支持 NVIDIA Hopper（H100/H200）、Ada Lovelace（L40S、RTX 4090）和 AMD MI300。Ampere GPU（A100）支持 FP8 存储但缺乏完整的计算加速。

量化工作流程为：将 Key/Value 张量以 FP8 格式存储，配合 per-tensor FP32 缩放因子 → 在注意力计算时反量化为 BF16/FP16。Hopper 上的 FlashAttention-3 支持完整的 FP8 注意力（Q、K、V 全部使用 FP8）以获得额外性能提升。

```bash
# 标准 FP8 KV 部署
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95

# 使用动态缩放因子计算
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-dtype fp8 \
    --calculate-kv-scales
```

**关键限制**：FlashAttention-2 不支持 FP8 KV cache；vLLM 会回退到 XFormers。推荐使用 FlashInfer 后端以获得最佳 FP8 KV 性能。

-----

## 精度损失在正确校准下很小

大量 benchmark 测试表明，经过正确校准的 FP8 KV cache 通常可实现 **<1% 的精度下降**：

|Benchmark           |FP8 vs FP16/BF16 影响|备注                        |
|--------------------|-------------------|--------------------------|
|**MMLU**            |<0.5% 损失           |大多数模型几乎相同                 |
|**GSM8K**           |<1% 损失             |思维链（CoT）保持完好              |
|**LiveCodeBench**   |~0% 损失             |代码生成鲁棒                    |
|**MBPP**            |<1.5% 损失           |语法退化很小                    |
|**Ruler 64K（长上下文）** |0.1% 损失            |95.5% vs 95.6%（Qwen3-480B）|
|**C-Eval/CMMLU（中文）**|稳定                 |核心语义保持                    |

**长上下文性能**保持稳健。NVIDIA 在 Ruler 64K 上的测试——这是一个”量化噪声通常会累积”的设置——显示 FP8 在 Qwen3-480B-A35B 上保持 **95.5%** 准确率，而 FP16 为 **95.6%**。大海捞针测试表明 ShadowKV “即使在 FP8 精度下也能保持准确性并实现始终如一的高性能”。

**数学推理**需要特别注意。研究表明 Key 张量”对量化噪声本质上比 Value 更敏感”。对于数学密集型工作负载，混合精度配置如 K8V4（8-bit keys，4-bit values）优于统一的 K4V4。纯 FP8（8-bit）保持接近基线的 GSM8K：使用校准 scales 的 Llama-3-8B-Instruct 达到 **77.48% 精确匹配**。

-----

## 校准缩放因子对生产精度至关重要

校准与未校准 FP8 KV cache 之间的精度差距显著：

|方法                           |相对精度损失   |运行时开销    |
|-----------------------------|---------|---------|
|**校准 scales**                |~0.2%    |无（预计算）   |
|**动态（`calculate_kv_scales`）**|~0.2-0.5%|首次计算有轻微开销|
|**未校准（scale=1.0）**           |~1-2%    |无        |

校准使用代表性数据（512+ 样本）计算每层最优的 k_scale 和 v_scale 因子：

```python
from llmcompressor import oneshot

# LLM Compressor FP8 KV cache 校准配置
recipe = """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore: ["lm_head"]
      config_groups:
        group_0:
          weights: {num_bits: 8, type: float, strategy: tensor, dynamic: false, symmetric: true}
          input_activations: {num_bits: 8, type: float, strategy: tensor, dynamic: false, symmetric: true}
          targets: ["Linear"]
      kv_cache_scheme:
        num_bits: 8
        type: float
        strategy: tensor
        dynamic: false
        symmetric: true
"""

oneshot(model=model, dataset=calibration_ds, recipe=recipe, 
        max_seq_length=2048, num_calibration_samples=512)
```

校准过程观察每层 K 和 V 投影的 min/max 激活值，计算：`scale = max_fp8_range / max(|activation_values|)`。**8B 模型需要 5-10 分钟**，**70B 模型使用多 GPU 张量并行需要 30-60 分钟**。

-----

## 针对特定模型的部署建议

**Qwen3-Next-80B-A3B**：

- 使用官方 FP8 权重 checkpoint（`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`）
- 设置 `--kv-cache-dtype auto`（绝不使用 `fp8`）
- 需要 vLLM nightly 版本
- SGLang 目前可能比 vLLM 更稳定

**GPT-oss-120B**：

- 使用原生 MXFP4 权重（模型可在单张 80GB GPU 上运行）
- 设置 `--kv-cache-dtype auto`
- FP8 KV cache 需等待未来 vLLM 版本的内核修复

**标准架构模型（Llama-3.1、Qwen3-235B）**：

- 使用 LLM Compressor 创建校准 checkpoint
- 使用 FlashInfer 后端以获得最佳 FP8 KV 性能
- 预期 ~50% KV cache 显存减少，~2x 上下文长度能力
- 生产部署前用 MMLU/GSM8K 验证

|模型                |FP8 权重    |FP8 KV Cache|推荐配置            |
|------------------|----------|------------|----------------|
|Qwen3-Next-80B-A3B|✅ 官方支持    |❌ 不可用       |FP8 权重 + auto KV|
|GPT-oss-120B      |❌ 原生 MXFP4|❌ 不可用       |仅使用原生量化         |
|Qwen3-235B-A22B   |✅ 官方支持    |⚠️ 需谨慎测试     |FP8 权重 + 校准 KV  |
|Llama-3.1-70B     |✅ 支持      |✅ 完全支持      |FP8 权重 + 校准 KV  |

-----

## 结论

最新的混合注意力模型（Qwen3-Next、GPT-oss-120B）代表了一个前沿领域，**架构创新超前于量化工具链**。它们的新型注意力机制——注意力 sink、滑动窗口、线性注意力变体——打破了当前 FP8 KV cache 内核中的假设。对于这些模型，显存优化路径是通过权重量化（FP8 或 MXFP4），而非 KV cache 压缩。

对于传统 Transformer 架构，FP8 KV cache 仍然非常有效：使用校准缩放因子可实现 **<1% 精度损失** 和 **2x 显存减少**。关键洞见是**校准很重要**——0.2% 和 2% 精度损失之间的差异足以证明 30-60 分钟校准开销对于生产工作负载是值得的。动态缩放（`calculate_kv_scales=True`）为快速实验提供了一个可靠的折中方案，无需完整的校准流程。
