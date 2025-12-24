import asyncio
import time
import json
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
import numpy as np
from typing import Optional

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
    input_len: int,
    output_len: int,
) -> dict:
    """单次请求，返回计时信息"""
    # 生成随机 prompt（简化版）
    prompt = "Hello " * (input_len // 2)
    
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
        
        return {
            "success": True,
            "latency": end_time - start_time,
            "ttft": (first_token_time - start_time) if first_token_time else None,
            "tokens": tokens_generated,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

async def run_concurrent_requests(
    model: str,
    input_len: int,
    output_len: int,
    num_requests: int,
    concurrency: int,
) -> BenchmarkResult:
    """并发测试"""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def limited_request():
        async with semaphore:
            return await single_request(model, input_len, output_len)
    
    start = time.perf_counter()
    results = await asyncio.gather(*[limited_request() for _ in range(num_requests)])
    total_time = time.perf_counter() - start
    
    # 统计
    successful = [r for r in results if r["success"]]
    latencies = [r["latency"] for r in successful]
    ttfts = [r["ttft"] for r in successful if r["ttft"]]
    total_tokens = sum(r["tokens"] for r in successful)
    
    return BenchmarkResult(
        concurrency=concurrency,
        context_length=input_len,
        num_requests=num_requests,
        total_time=total_time,
        throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
        avg_latency=np.mean(latencies) if latencies else 0,
        p50_latency=np.percentile(latencies, 50) if latencies else 0,
        p99_latency=np.percentile(latencies, 99) if latencies else 0,
        ttft_avg=np.mean(ttfts) if ttfts else 0,
        success_rate=len(successful) / num_requests,
    )

async def full_benchmark(
    model: str,
    concurrency_levels: list[int],
    context_lengths: list[int],
    output_len: int = 256,
    num_requests: int = 50,
) -> list[BenchmarkResult]:
    """完整矩阵测试"""
    results = []
    
    for ctx_len in context_lengths:
        for conc in concurrency_levels:
            print(f"Testing: context={ctx_len}, concurrency={conc}")
            result = await run_concurrent_requests(
                model=model,
                input_len=ctx_len,
                output_len=output_len,
                num_requests=num_requests,
                concurrency=conc,
            )
            results.append(result)
            print(f"  → Throughput: {result.throughput_tokens_per_sec:.1f} tok/s")
    
    return results

# 运行测试
async def main():
    results = await full_benchmark(
        model="your-model-name",
        concurrency_levels=[1, 2, 5, 10],
        context_lengths=[1024, 10240, 32768, 65536],
        num_requests=30,
    )
    
    # 保存结果
    with open("benchmark_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    return results

if __name__ == "__main__":
    results = asyncio.run(main())