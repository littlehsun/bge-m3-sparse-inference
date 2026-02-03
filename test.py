#!/usr/bin/env python3
"""Test script for BGE-M3 Sparse Embedding Service"""

import argparse
import json
import time
import requests
from typing import List

DEFAULT_URL = "http://localhost:8080"


def test_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        result = r.json()
        print(f"✅ Health: {result}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_info(base_url: str):
    try:
        r = requests.get(f"{base_url}/info", timeout=10)
        result = r.json()
        print(f"✅ Info: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"❌ Info failed: {e}")


def test_sparse_single(base_url: str):
    try:
        r = requests.post(
            f"{base_url}/embed_sparse",
            json={"inputs": "Hello world, this is a test for sparse embeddings."},
            timeout=30,
        )
        result = r.json()
        print(f"✅ Sparse single: {len(result[0])} tokens")
        if result[0]:
            print(f"   Sample: {result[0][:3]}")
    except Exception as e:
        print(f"❌ Sparse single failed: {e}")


def test_sparse_batch(base_url: str, batch_size: int = 32) -> float:
    texts = [f"Test sentence {i} for benchmarking sparse embeddings." for i in range(batch_size)]
    
    try:
        start = time.perf_counter()
        r = requests.post(
            f"{base_url}/embed_sparse",
            json={"inputs": texts},
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        
        result = r.json()
        throughput = batch_size / elapsed
        print(f"✅ Batch {batch_size}: {elapsed*1000:.1f}ms ({throughput:.1f} texts/sec)")
        return elapsed
    except Exception as e:
        print(f"❌ Batch {batch_size} failed: {e}")
        return 0


def benchmark(base_url: str):
    print("\n" + "=" * 50)
    print("BENCHMARK - Sparse Embeddings")
    print("=" * 50)
    
    for batch_size in [1, 8, 32, 64, 128]:
        # Warmup
        test_sparse_batch(base_url, batch_size)
        # Measure
        times = [test_sparse_batch(base_url, batch_size) for _ in range(3)]
        avg = sum(t for t in times if t > 0) / max(1, len([t for t in times if t > 0]))
        print(f"   → Avg: {avg*1000:.1f}ms ({batch_size/avg:.1f} texts/sec)")


def main():
    parser = argparse.ArgumentParser(description="Test BGE-M3 Sparse Service")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args = parser.parse_args()
    
    print(f"\n🔍 Testing at {args.url}\n")
    
    if not test_health(args.url):
        print("\n⚠️  Server not available")
        return
    
    test_info(args.url)
    print()
    test_sparse_single(args.url)
    test_sparse_batch(args.url, 32)
    
    if args.benchmark:
        benchmark(args.url)
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
