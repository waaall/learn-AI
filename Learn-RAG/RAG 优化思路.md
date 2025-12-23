

二、RAG 调用侧的并发与性能优化点

  

  

你之前说“平均两次 embedding + 五次 LLM，一次几千 token”。真正的性能瓶颈往往不是 vLLM 本身，而是你把“长 prefill”重复做了 5 遍。下面这些改动通常比继续拧 vLLM 参数更有效。

  

  

1) 先从结构上把 5 次 LLM 压到 1~2 次

  

  

高并发时，最贵的是长输入 prefill。常见合并方式：

  

- 把“query 改写、检索指令生成、证据选择、最终回答（含引用格式）”尽量合成一次模型调用
- 或最多两次：一次生成检索 query/过滤条件；一次拿检索结果生成最终答案  
    这样你等价于把每轮对话的“几千 token prefill”从 5 次降到 1~2 次，吞吐和尾延迟通常是数量级改善。

  

  

  

2) 让 APC 真正命中：把“稳定前缀”做成强约束

  

  

APC 的前提是新请求与旧请求共享相同前缀 KV block。

因此你在 RAG 侧要做到：

  

- 系统提示词、角色说明、输出格式、工具 schema：尽量固定不变
- 把高度动态的部分（用户问题、检索片段）放在 prompt 后半段
- 检索片段的拼接格式要稳定：同样的分隔符、同样的字段顺序、同样的引用模板
- 不要每次请求都随机插入时间戳、随机 trace id 到系统提示词里（这会破坏前缀一致性）

  

  

如果你以后跑多副本：需要尽量把同前缀请求路由到同一副本，否则缓存命中会被稀释。Ray 的 prefix-aware routing 文档就是围绕这个问题设计的。

  

  

3) 控制“检索带来的 token 洪水”，优先减少 prefill

  

  

你现在“每次几千 token”，通常来自：topK 太大、chunk 太长、重复 chunk、无效上下文。建议优先做：

  

- 检索：更激进的去重与聚合（同文档相邻 chunk 合并/裁剪）
- rerank：强制只保留 N 个证据（比如 6~12 个短 chunk，而不是 30 个长 chunk）
- 上下文压缩：对每个 chunk 做摘要/要点提取再送大模型（可用更小的模型或规则）
- 预算化：在拼 prompt 前做 token 预算器，硬性卡住“证据区最大 token”

  

  

这一步会同时让 vLLM 的 chunked prefill 更有效（长 prompt 变少）、APC 更容易命中（固定前缀占比更高）。

  

  

4) 并发策略：RAG 编排层做“背压 + 批处理”

  

  

- embedding：把多用户同时到来的 query 合并 batch（embedding 模型吞吐会更好）
- 向量库：异步 IO + 连接池 + 超时；对热门 query 做短 TTL 缓存
- LLM：应用层做并发上限与排队，不要无限并发把 vLLM 推到 preemption/recompute（文档已经说明这会伤端到端延迟）。 

  

  

  

5) 用指标把瓶颈定性，否则容易“盲调”

  

  

建议你把每轮对话拆成四段打点：

  

- embedding1/2：耗时、批大小
- retrieval：topK、返回 token 数
- LLM：TTFT、TPOT、输入/输出 token  
    然后对照 vLLM 的 /metrics 里的 KV 使用率、prefix cache 命中率、请求延迟直方图一起看，vLLM 官方指标设计文档就按这种可观测性思路写的。 

  

  

  

  

  

给你一个“优先级路线图”

  

  

1. vLLM：>=0.10.2 + 开 APC + 开 chunked prefill + 先把 preemption 打掉（靠 gpu_memory_utilization / max_num_seqs / max_num_batched_tokens）。 
2. RAG：把 5 次 LLM 压到 1~2 次；同时做 prompt 预算与证据裁剪，让“几千 token”下降。
3. 监控：上 Prometheus/Grafana，盯 KV 使用与 prefix cache 命中，形成可重复的压测-调参闭环。 
4. 仍然被长 prompt 干扰：再考虑 vLLM 的 Prefill/Decode 解耦（experimental）。 

  

  

如果你把两组数据给我（真实分布，不用很精确）：

  

- 输入 token 的 p50/p95、输出 token 的 p50/p95
- 峰值并发时“同时在飞”的 LLM 请求数  
    我可以按你的目标（更偏吞吐还是更偏 p95）给一套更具体的 vLLM 参数区间与压测指标阈值。