# VLLM issues

## GPU memory utilization not work

搜了一圈 vLLM 官方文档、GitHub issue / discussion，结论是：`--gpu-memory-utilization`不是“硬性限制 nvidia-smi 里这个进程最多占多少显存”的参数。

它是：vLLM 用来初始化 / 推算 KV cache 池大小的预算参数，而不是 Docker / CUDA 层面的显存上限。

官方 CLI 文档里写得比较关键：`--gpu-memory-utilization`是 “model executor” 使用 GPU 显存的比例，而且是per-instance limit，只作用于当前 vLLM 实例；如果同一张卡上跑两个 vLLM，每个都可以设置 0.5，它们并不会互相感知。([vLLM](https://docs.vllm.ai/en/v0.19.1/cli/serve/ "vllm serve - vLLM"))
官方优化文档也说，vLLM 会用这个比例pre-allocate GPU cache，调大它是为了给 KV cache 更多空间；如果 KV cache 不够，建议降低`max_num_seqs`或`max_num_batched_tokens`。([vLLM](https://docs.vllm.ai/en/stable/configuration/optimization/ "Optimization and Tuning - vLLM"))

所以它不是这个意思：

```text
gpu-memory-utilization=0.60
=> nvidia-smi 里该进程永远 <= 总显存 * 0.60
```

更接近这个意思：

```text
gpu-memory-utilization=0.60
=> vLLM 在启动 profiling 时，按这个比例估算自己可用于 model executor / KV cache 的预算
=> 然后推算 GPU KV cache blocks / tokens
=> 但 CUDA context、runtime allocator、JIT、Triton、CUDA graph、后续请求触发的缓存等，不一定严格卡在这个数下面
```

GitHub 上也有不少人困惑这个参数。有用户提 feature request，希望 vLLM 增加“限制总 GPU 显存”的参数，因为即使用了`--max-model-len`、`--max-num-batched-tokens`、`--max-num-seqs`等参数，剩余显存还是可能被 KV cache 吃掉；这个 issue 是“Limit total GPU memory”，最后是 closed as not planned。([GitHub](https://github.com/vllm-project/vllm/issues/20256 "[Feature]: Limit total GPU memory · Issue #20256 · vllm-project/vllm · GitHub"))
还有用户报告`gpu_memory_utilization=0.9`时实际显存使用率到 0.99，issue 标题就是 “The parameter gpu_memory_utilization does not take effect”。([GitHub](https://github.com/vllm-project/vllm/issues/10637 "[Bug]:The parameter gpu_memory_utilization does not take effect · Issue #10637 · vllm-project/vllm · GitHub"))
也有人反过来报告设置 0.9 但 vLLM 没有吃到 90%，说明这个参数也不是“必须预留到这个比例”。([GitHub](https://github.com/vllm-project/vllm/issues/7768 "[Bug]: gpu-memory-utilization does not pickup enough GPU memory · Issue #7768 · vllm-project/vllm · GitHub"))
embedding 模型上也有人遇到类似困惑：设了`--gpu-memory-utilization 0.25`，以为会吃 4GB，但实际只看到 1.5GB。([GitHub](https://github.com/vllm-project/vllm/issues/12308 "[Bug]: Possible GPU Memory Utilization issue/bug for embeddings model · Issue #12308 · vllm-project/vllm · GitHub"))

回到的机器，日志里已经能印证这一点。的 LLM 当前配置是：

```yaml
--max-model-len 81920
--max-num-seqs 8
--max-num-batched-tokens 65536
--gpu-memory-utilization 0.60
```

这个配置确实生效了。
启动 profiling 时，vLLM 算出来的是：

```text
Available KV cache memory: 11.7 GiB
GPU KV cache size: 565,638 tokens
Maximum concurrency for 81,920 tokens per request: 6.90x
```

这说明`0.60`主要影响的是KV cache 预算，但后来看到`nvidia-smi`里 LLM 从约 51GB 涨到 59.56GB，这部分很可能是运行期高水位：Triton JIT、torch.compile、allocator cache、MoE kernel workspace、prefix / mamba cache、真实请求 shape 触发的额外缓存等。的日志里也确实有 “Triton kernel JIT compilation during inference” 的多条记录。

所以现在更靠谱的做法不是继续用`gpu-memory-utilization`猜，而是直接指定 KV cache 大小。

官方文档里有一个更精确的参数：

```bash
--kv-cache-memory-bytes
```

它的说明是：直接指定每张 GPU 上 KV cache 的大小；如果设置了它，vLLM 会忽略`gpu_memory_utilization`，所以它比百分比更细。([vLLM](https://docs.vllm.ai/en/v0.19.1/cli/serve/ "vllm serve - vLLM"))

按现在日志倒推：

```text
11.7 GiB KV cache = 565,638 tokens
81920 tokens × 4 = 327,680 tokens
需要 KV cache ≈ 11.7 × 327680 / 565638 ≈ 6.78 GiB
```

所以如果想保留80K 上下文 × 4 并发，建议直接这样配：

```yaml
# 上下文 + 并发设置
- "--max-model-len"
- "81920"

- "--max-num-seqs"
- "4"

- "--max-num-batched-tokens"
- "65536"

# 直接限制 KV cache，大约 8GiB，够 80K * 4，留一点余量
- "--kv-cache-memory-bytes"
- "8589934592"
```

然后可以把这个删掉或保留但知道它会被忽略：

```yaml
- "--gpu-memory-utilization"
- "0.54"
```

我建议删掉，避免误导自己：

```yaml
# 不再使用 --gpu-memory-utilization，改用 --kv-cache-memory-bytes
```

如果想再激进一点释放显存，可以试：

```yaml
- "--kv-cache-memory-bytes"
- "7516192768"   # 7GiB
```

但 7GiB 对`81920 × 4`比较贴边，可能刚好不够。更稳的是`8GiB`。

重建：

```bash
docker compose -f compose-llm.yml up -d --force-recreate
```

然后看启动日志：

```bash
docker logs vllm-qwen36-35b-a3b 2>&1 | \
grep -Ei 'kv cache|Available KV cache|Maximum concurrency|Model loading took|gpu_memory_utilization|kv-cache-memory'
```

要看到类似：

```text
GPU KV cache size: ...
Maximum concurrency for 81,920 tokens per request: >= 4.0x
```

结论就是：

```text
--gpu-memory-utilization 不是真正的显存硬限制。
想控制 vLLM 显存，尤其是 KV cache，优先用 --kv-cache-memory-bytes。
场景建议：max-num-seqs=4 + kv-cache-memory-bytes=8589934592。
```

这比`--gpu-memory-utilization 0.54`可控得多。