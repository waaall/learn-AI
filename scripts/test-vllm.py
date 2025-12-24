import asyncio
import time
import json
import logging
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
import numpy as np
from typing import Optional
from datetime import datetime
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'vllm_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ShareGPT风格的真实对话数据集
SHAREGPT_SAMPLES = [
    {
        "conversations": [
            {"role": "user", "content": "Can you explain what machine learning is in simple terms?"},
            {"role": "assistant", "content": "Machine learning is a branch of artificial intelligence where computers learn from data without being explicitly programmed..."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Write a Python function to calculate the Fibonacci sequence up to n terms."},
            {"role": "assistant", "content": "Here's a Python function that calculates the Fibonacci sequence..."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "What are the main differences between SQL and NoSQL databases? When would you use each?"},
            {"role": "assistant", "content": "SQL and NoSQL databases differ in several key ways..."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "I'm planning a trip to Japan. Can you suggest a 7-day itinerary covering Tokyo and Kyoto?"},
            {"role": "assistant", "content": "Here's a wonderful 7-day itinerary for Japan..."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Explain the concept of blockchain technology and its potential applications beyond cryptocurrency."},
            {"role": "assistant", "content": "Blockchain is a distributed ledger technology..."}
        ]
    },
]

def load_sharegpt_data(file_path: Optional[str] = None) -> list[dict]:
    """加载ShareGPT数据集，如果没有文件则使用内置样本"""
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"从 {file_path} 加载了 {len(data)} 条对话")
                return data
        except Exception as e:
            logger.warning(f"无法加载文件 {file_path}: {e}，使用内置样本")

    logger.info(f"使用内置 {len(SHAREGPT_SAMPLES)} 条对话样本")
    return SHAREGPT_SAMPLES

@dataclass
class BenchmarkResult:
    concurrency: int
    context_length: int
    num_requests: int
    total_time: float
    throughput_tokens_per_sec: float
    avg_latency: float
    p50_latency: float
    p99_latency: float
    ttft_avg: float  # Time to First Token
    success_rate: float
    
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

async def single_request(
    model: str,
    prompt: str,
    output_len: int,
    request_id: int = 0,
) -> dict:
    """单次请求，返回计时信息"""
    start_time = time.perf_counter()
    first_token_time = None
    tokens_generated = 0

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=output_len,
            stream=True,
        )

        async for chunk in stream:
            if first_token_time is None and chunk.choices[0].delta.content:
                first_token_time = time.perf_counter()
            if chunk.choices[0].delta.content:
                tokens_generated += 1

        end_time = time.perf_counter()

        result = {
            "success": True,
            "latency": end_time - start_time,
            "ttft": (first_token_time - start_time) if first_token_time else None,
            "tokens": tokens_generated,
        }
        logger.debug(f"请求 {request_id} 成功: {tokens_generated} tokens, 延迟 {result['latency']:.2f}s")
        return result

    except Exception as e:
        error_msg = f"请求 {request_id} 失败: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "request_id": request_id,
        }

async def run_concurrent_requests(
    model: str,
    prompts: list[str],
    output_len: int,
    num_requests: int,
    concurrency: int,
) -> BenchmarkResult:
    """并发测试"""
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(request_id: int):
        async with semaphore:
            # 循环使用prompts
            prompt = prompts[request_id % len(prompts)]
            return await single_request(model, prompt, output_len, request_id)

    logger.info(f"开始并发测试: 并发数={concurrency}, 请求数={num_requests}")
    start = time.perf_counter()
    results = await asyncio.gather(*[limited_request(i) for i in range(num_requests)])
    total_time = time.perf_counter() - start

    # 统计
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in successful]
    ttfts = [r["ttft"] for r in successful if r["ttft"]]
    total_tokens = sum(r["tokens"] for r in successful)

    # 记录错误详情
    if failed:
        logger.warning(f"有 {len(failed)} 个请求失败:")
        error_types = {}
        for r in failed:
            error_type = r.get("error_type", "Unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        for error_type, count in error_types.items():
            logger.warning(f"  - {error_type}: {count} 次")

    result = BenchmarkResult(
        concurrency=concurrency,
        context_length=len(prompts[0]) if prompts else 0,  # 使用第一个prompt的长度作为参考
        num_requests=num_requests,
        total_time=total_time,
        throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
        avg_latency=np.mean(latencies) if latencies else 0,
        p50_latency=np.percentile(latencies, 50) if latencies else 0,
        p99_latency=np.percentile(latencies, 99) if latencies else 0,
        ttft_avg=np.mean(ttfts) if ttfts else 0,
        success_rate=len(successful) / num_requests if num_requests > 0 else 0,
    )

    logger.info(f"测试完成: 成功率={result.success_rate*100:.1f}%, 吞吐量={result.throughput_tokens_per_sec:.1f} tok/s")
    return result

def generate_prompts_from_sharegpt(
    sharegpt_data: list[dict],
    target_length: int,
    count: int = 10
) -> list[str]:
    """从ShareGPT数据生成指定长度的prompts"""
    prompts = []

    for _ in range(count):
        # 随机选择一个对话
        conversation = random.choice(sharegpt_data)
        user_msg = conversation["conversations"][0]["content"]

        # 根据目标长度调整prompt
        current_len = len(user_msg)
        if current_len < target_length:
            # 重复内容来达到目标长度
            repeat_times = (target_length // current_len) + 1
            prompt = (user_msg + " ") * repeat_times
            prompt = prompt[:target_length]
        else:
            # 截取到目标长度
            prompt = user_msg[:target_length]

        prompts.append(prompt)

    return prompts

async def warmup(model: str, num_warmup: int = 5):
    """预热阶段，发送几个请求让模型加载"""
    logger.info(f"开始预热阶段，发送 {num_warmup} 个请求...")
    warmup_prompts = ["Hello, how are you?"] * num_warmup

    for i in range(num_warmup):
        result = await single_request(
            model=model,
            prompt=warmup_prompts[i],
            output_len=50,
            request_id=f"warmup_{i}"
        )
        if result["success"]:
            logger.info(f"预热请求 {i+1}/{num_warmup} 完成")
        else:
            logger.warning(f"预热请求 {i+1}/{num_warmup} 失败: {result.get('error', 'Unknown')}")

    logger.info("预热阶段完成\n")

async def full_benchmark(
    model: str,
    concurrency_levels: list[int],
    context_lengths: list[int],
    output_len: int = 256,
    num_requests: int = 50,
    sharegpt_data: Optional[list[dict]] = None,
) -> list[BenchmarkResult]:
    """完整矩阵测试"""
    results = []

    # 加载数据集
    if sharegpt_data is None:
        sharegpt_data = load_sharegpt_data()

    for ctx_len in context_lengths:
        # 为当前上下文长度生成prompts
        prompts = generate_prompts_from_sharegpt(sharegpt_data, ctx_len, count=20)
        logger.info(f"\n{'='*60}")
        logger.info(f"测试上下文长度: {ctx_len} 字符")
        logger.info(f"{'='*60}")

        for conc in concurrency_levels:
            result = await run_concurrent_requests(
                model=model,
                prompts=prompts,
                output_len=output_len,
                num_requests=num_requests,
                concurrency=conc,
            )
            results.append(result)
            print(f"  → 吞吐量: {result.throughput_tokens_per_sec:.1f} tok/s, "
                  f"平均延迟: {result.avg_latency:.2f}s, "
                  f"TTFT: {result.ttft_avg:.3f}s")

    return results

# 运行测试
async def main(
    model: str = "your-model-name",
    sharegpt_file: Optional[str] = None,
    enable_warmup: bool = True,
):
    """主测试流程"""
    logger.info("="*60)
    logger.info("vLLM 性能基准测试开始")
    logger.info("="*60)

    # 加载数据集
    sharegpt_data = load_sharegpt_data(sharegpt_file)

    # 预热阶段
    if enable_warmup:
        await warmup(model, num_warmup=5)

    # 执行完整基准测试
    results = await full_benchmark(
        model=model,
        concurrency_levels=[1, 2, 5, 10],
        context_lengths=[500, 2000, 5000],  # 使用字符长度而非token
        num_requests=30,
        sharegpt_data=sharegpt_data,
    )

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    logger.info(f"\n测试完成！结果已保存到: {output_file}")

    # 打印汇总
    logger.info("\n" + "="*60)
    logger.info("测试汇总")
    logger.info("="*60)
    for r in results:
        logger.info(f"并发={r.concurrency}, 上下文={r.context_length}: "
                   f"吞吐量={r.throughput_tokens_per_sec:.1f} tok/s, "
                   f"成功率={r.success_rate*100:.1f}%")

    return results

if __name__ == "__main__":
    # 可以通过修改这里的参数来自定义测试
    results = asyncio.run(main(
        model="your-model-name",  # 修改为实际模型名称
        sharegpt_file=None,  # 可选：指定ShareGPT JSON文件路径
        enable_warmup=True,
    ))