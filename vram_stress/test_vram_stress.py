"""
VRAM Stress Test for Sparse Embedding API

This script tests GPU memory management by:
1. Reading test1.py ~ test9.py (PDF to MD content)
2. Chunking text by ~2000 tokens
3. Sending batch requests to sparse embedding API
4. Monitoring GPU VRAM usage via API
"""

import os
import time
import requests
from typing import List
from pathlib import Path


# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8080")
EMBED_SPARSE_ENDPOINT = f"{API_URL}/embed_sparse"
GPU_MEMORY_ENDPOINT = f"{API_URL}/gpu_memory"
CHUNK_SIZE_CHARS = 6000  # ~2000 tokens (assuming ~3 chars per token for mixed content)
BATCH_SIZE = 64  # Number of chunks per API request
TEST_FILES = [f"test{i}.py" for i in range(1, 10)]


def get_gpu_memory() -> dict:
    """Get current GPU memory usage from server API"""
    try:
        response = requests.get(GPU_MEMORY_ENDPOINT, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            "allocated_gb": data.get("allocated_gb", -1),
            "reserved_gb": data.get("reserved_gb", -1),
            "max_allocated_gb": data.get("max_allocated_gb", -1),
        }
    except Exception:
        return {"allocated_gb": -1, "reserved_gb": -1, "max_allocated_gb": -1}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS) -> List[str]:
    """Split text into chunks of approximately chunk_size characters"""
    chunks = []

    # Split by paragraphs first to avoid breaking mid-sentence
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # If single paragraph is too long, split by sentences
            if len(para) > chunk_size:
                words = para.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        current_chunk += word + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = word + " "
            else:
                current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def load_test_files(base_dir: str) -> List[str]:
    """Load all test files and return list of all chunks"""
    all_chunks = []

    for filename in TEST_FILES:
        filepath = Path(base_dir) / filename
        if filepath.exists():
            print(f"Loading {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = chunk_text(content)
            all_chunks.extend(chunks)
            print(f"  - {len(content):,} chars -> {len(chunks)} chunks")
        else:
            print(f"  - {filename} not found, skipping")

    return all_chunks


def call_sparse_embedding(texts: List[str], truncate: bool = True) -> dict:
    """Call sparse embedding API and return response with timing"""
    start_time = time.perf_counter()

    try:
        response = requests.post(
            EMBED_SPARSE_ENDPOINT,
            json={"inputs": texts, "truncate": truncate},
            timeout=300  # 5 minutes timeout for large batches
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - start_time

        return {
            "success": True,
            "elapsed_ms": elapsed * 1000,
            "num_results": len(response.json()),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {
            "success": False,
            "elapsed_ms": elapsed * 1000,
            "error": str(e),
        }


def run_stress_test():
    """Run the VRAM stress test"""
    print("=" * 60)
    print("VRAM Stress Test for Sparse Embedding API")
    print("=" * 60)

    # Get script directory
    script_dir = Path(__file__).parent

    # Check API health
    print(f"\nChecking API at {API_URL}...")
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
        health.raise_for_status()
        print("API is healthy!")
    except Exception as e:
        print(f"ERROR: API not reachable - {e}")
        print("Please start the API server first: python main.py")
        return

    # Load test files
    print("\n" + "-" * 60)
    print("Loading test files...")
    print("-" * 60)
    all_chunks = load_test_files(script_dir)

    if not all_chunks:
        print("ERROR: No test files found!")
        return

    print(f"\nTotal chunks to process: {len(all_chunks)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {(len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE}")

    # Initial GPU memory (from server API)
    initial_mem = get_gpu_memory()
    if initial_mem['allocated_gb'] >= 0:
        print(f"\nInitial GPU Memory: Allocated={initial_mem['allocated_gb']:.2f}GB, Reserved={initial_mem['reserved_gb']:.2f}GB")
    else:
        print("\n(GPU memory API not available)")

    # Run batch requests
    print("\n" + "-" * 60)
    print("Running batch requests...")
    print("-" * 60)

    total_start = time.perf_counter()
    batch_num = 0
    total_processed = 0
    total_errors = 0

    # Track memory over time
    memory_samples = []

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_num += 1
        batch_chunks = all_chunks[i:i + BATCH_SIZE]

        result = call_sparse_embedding(batch_chunks)

        if result["success"]:
            total_processed += len(batch_chunks)
            status = "OK"
        else:
            total_errors += 1
            status = f"ERROR: {result.get('error', 'unknown')}"

        # Get current GPU memory
        current_mem = get_gpu_memory()
        memory_samples.append(current_mem)

        # Print progress
        if current_mem['allocated_gb'] >= 0:
            print(f"Batch {batch_num:3d}: {len(batch_chunks):3d} chunks | "
                  f"{result['elapsed_ms']:7.1f}ms | "
                  f"GPU: {current_mem['allocated_gb']:.2f}GB alloc, {current_mem['reserved_gb']:.2f}GB reserved | "
                  f"{status}")
        else:
            print(f"Batch {batch_num:3d}: {len(batch_chunks):3d} chunks | "
                  f"{result['elapsed_ms']:7.1f}ms | {status}")

    total_elapsed = time.perf_counter() - total_start

    # Final summary
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"Total chunks processed: {total_processed}")
    print(f"Total batches: {batch_num}")
    print(f"Total errors: {total_errors}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Average per chunk: {total_elapsed / max(total_processed, 1) * 1000:.2f}ms")

    if memory_samples and memory_samples[0]['allocated_gb'] >= 0:
        final_mem = get_gpu_memory()
        max_allocated = max(s['allocated_gb'] for s in memory_samples)
        max_reserved = max(s['reserved_gb'] for s in memory_samples)

        print(f"\nGPU Memory Analysis:")
        print(f"  Initial:  Allocated={initial_mem['allocated_gb']:.2f}GB, Reserved={initial_mem['reserved_gb']:.2f}GB")
        print(f"  Peak:     Allocated={max_allocated:.2f}GB, Reserved={max_reserved:.2f}GB")
        print(f"  Final:    Allocated={final_mem['allocated_gb']:.2f}GB, Reserved={final_mem['reserved_gb']:.2f}GB")
        print(f"  Growth:   Allocated={final_mem['allocated_gb'] - initial_mem['allocated_gb']:+.2f}GB, "
              f"Reserved={final_mem['reserved_gb'] - initial_mem['reserved_gb']:+.2f}GB")

        # Check for memory leak
        growth = final_mem['reserved_gb'] - initial_mem['reserved_gb']
        if growth > 1.0:
            print(f"\n  WARNING: Significant memory growth detected ({growth:.2f}GB)!")
            print("  This may indicate a memory leak.")
        else:
            print(f"\n  Memory growth is acceptable ({growth:.2f}GB)")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_stress_test()
