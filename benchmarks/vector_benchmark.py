#!/usr/bin/env python3
"""
Vector Search Benchmark — NumPy vs FAISS.
Compares add latency, search latency, and memory usage at 1K, 10K, 100K scales.

Usage:
    python benchmarks/vector_benchmark.py
"""

import gc
import sys
import time
import tracemalloc
import numpy as np


# ── Helpers ──────────────────────────────────────────────

def generate_vectors(n: int, dim: int = 384) -> np.ndarray:
    """Generate n random normalized vectors."""
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1_000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def fmt_mem(bytes_used: int) -> str:
    if bytes_used < 1024:
        return f"{bytes_used} B"
    elif bytes_used < 1024**2:
        return f"{bytes_used / 1024:.1f} KB"
    else:
        return f"{bytes_used / 1024**2:.1f} MB"


# ── Benchmark Functions ──────────────────────────────────

def benchmark_numpy(vectors: np.ndarray, queries: np.ndarray, top_k: int = 10):
    """Benchmark pure NumPy brute-force search."""
    n, dim = vectors.shape

    # Add
    tracemalloc.start()
    t0 = time.perf_counter()
    matrix = vectors.copy()  # Simulate storing
    add_time = time.perf_counter() - t0
    _, add_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Search (single query)
    search_times = []
    for q in queries:
        q = q.reshape(1, -1)
        t0 = time.perf_counter()
        sims = np.dot(matrix, q.T).flatten()
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        search_times.append(time.perf_counter() - t0)

    return {
        "backend": "NumPy",
        "n": n,
        "add_time": add_time,
        "add_memory": add_peak,
        "search_avg": np.mean(search_times),
        "search_p50": np.percentile(search_times, 50),
        "search_p95": np.percentile(search_times, 95),
    }


def benchmark_faiss(vectors: np.ndarray, queries: np.ndarray, top_k: int = 10):
    """Benchmark FAISS IndexFlatIP search."""
    try:
        import faiss
    except ImportError:
        return None

    n, dim = vectors.shape

    # Add
    tracemalloc.start()
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    add_time = time.perf_counter() - t0
    _, add_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Search (single query)
    search_times = []
    for q in queries:
        q = q.reshape(1, -1)
        t0 = time.perf_counter()
        scores, indices = index.search(q, top_k)
        search_times.append(time.perf_counter() - t0)

    return {
        "backend": "FAISS",
        "n": n,
        "add_time": add_time,
        "add_memory": add_peak,
        "search_avg": np.mean(search_times),
        "search_p50": np.percentile(search_times, 50),
        "search_p95": np.percentile(search_times, 95),
    }


# ── Main ─────────────────────────────────────────────────

def main():
    scales = [1_000, 10_000, 100_000]
    dim = 384
    n_queries = 100
    top_k = 10

    print("=" * 80)
    print("NEXUS Vector Search Benchmark — NumPy vs FAISS")
    print(f"Dimension: {dim}, Queries: {n_queries}, Top-K: {top_k}")
    print("=" * 80)
    print()

    all_results = []

    for n in scales:
        print(f"─── Scale: {n:,} vectors ───")

        vectors = generate_vectors(n, dim)
        queries = generate_vectors(n_queries, dim)

        gc.collect()
        np_result = benchmark_numpy(vectors, queries, top_k)
        all_results.append(np_result)

        gc.collect()
        faiss_result = benchmark_faiss(vectors, queries, top_k)
        if faiss_result:
            all_results.append(faiss_result)

        print(f"  NumPy:  add={fmt_time(np_result['add_time']):<10}  "
              f"search_avg={fmt_time(np_result['search_avg']):<10}  "
              f"search_p95={fmt_time(np_result['search_p95']):<10}  "
              f"mem={fmt_mem(np_result['add_memory'])}")
        if faiss_result:
            speedup = np_result["search_avg"] / faiss_result["search_avg"] if faiss_result["search_avg"] > 0 else 0
            print(f"  FAISS:  add={fmt_time(faiss_result['add_time']):<10}  "
                  f"search_avg={fmt_time(faiss_result['search_avg']):<10}  "
                  f"search_p95={fmt_time(faiss_result['search_p95']):<10}  "
                  f"mem={fmt_mem(faiss_result['add_memory'])}")
            print(f"  → FAISS is {speedup:.1f}x faster at search")
        else:
            print("  FAISS:  not installed — skipped")
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Backend':<8} {'Vectors':>10} {'Add':>12} {'Search Avg':>12} {'Search P95':>12} {'Memory':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['backend']:<8} {r['n']:>10,} {fmt_time(r['add_time']):>12} "
              f"{fmt_time(r['search_avg']):>12} {fmt_time(r['search_p95']):>12} "
              f"{fmt_mem(r['add_memory']):>10}")
    print()


if __name__ == "__main__":
    main()
