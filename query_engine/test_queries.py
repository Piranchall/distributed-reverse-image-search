"""
test_queries.py — Standalone test script for Task 3 (QueryProcessor)
Project 20: Distributed Reverse Image Search Engine — Milestone 2
Author: Mahnoor

Run from the repo root:
    python query_engine/test_queries.py

Expected output:
  - 500 to 5000 candidates per query
  - Average latency under 100 ms per query
  - Sync check output to share in group chat
"""

import os
import sys
import random
import time
import h5py
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from query_engine.query_processor import QueryProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────
INDEX_DIR = 'lsh_index'
DATA_DIR  = 'data'
CNN_PATH  = os.path.join(DATA_DIR, 'all_cnn_embeddings.h5')

N_TEST_QUERIES = 20

print("=" * 60)
print("Task 3 — QueryProcessor Standalone Test")
print("=" * 60)

# ── Instantiate ───────────────────────────────────────────────────────────────
qp = QueryProcessor(index_dir=INDEX_DIR, data_path=CNN_PATH)

# ── Pick 20 random image IDs ──────────────────────────────────────────────────
all_ids    = list(qp.id_to_index.keys())
test_ids   = random.sample(all_ids, N_TEST_QUERIES)

print(f"\nRunning {N_TEST_QUERIES} random queries ...\n")

candidate_counts = []

for i, image_id in enumerate(test_ids):
    # Prepare normalized embedding
    emb = qp.prepare_query_embedding(image_id=image_id)

    # Fan-out query
    t0         = time.time()
    candidates = qp.get_candidates(emb)
    elapsed_ms = (time.time() - t0) * 1000

    candidate_counts.append(len(candidates))
    print(f"  [{i+1:>2d}] ID={image_id:>7d} | "
          f"candidates={len(candidates):>5d} | "
          f"latency={elapsed_ms:>6.1f} ms")

# ── Summary ───────────────────────────────────────────────────────────────────
stats = qp.get_query_stats()

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  Queries run        : {stats['total_queries']}")
print(f"  Mean candidates    : {np.mean(candidate_counts):.0f}")
print(f"  Min  candidates    : {np.min(candidate_counts)}")
print(f"  Max  candidates    : {np.max(candidate_counts)}")
print(f"  Mean latency       : {stats['mean_latency_ms']:.1f} ms")
print(f"  P50  latency       : {stats['p50_latency_ms']:.1f} ms")
print(f"  P95  latency       : {stats['p95_latency_ms']:.1f} ms")
print(f"  P99  latency       : {stats['p99_latency_ms']:.1f} ms")

# ── Diagnostic warnings ───────────────────────────────────────────────────────
if np.mean(candidate_counts) < 200:
    print("\n  WARNING: Fewer than 200 candidates on average.")
    print("  HASH_SIZE may be too large — consider reducing it in lsh_config.json.")

if np.mean(candidate_counts) >= 5000:
    print("\n  WARNING: Hitting MAX_CANDIDATES cap on most queries.")
    print("  Consider increasing max_candidates in lsh_config.json for better recall.")

if stats['mean_latency_ms'] > 100:
    print("\n  WARNING: Mean latency > 100ms — target is under 100ms on a modern laptop.")

print("\n" + "=" * 60)
print("Sync check — share this block in group chat")
print("=" * 60)
print(f"  Index dir     : {os.path.abspath(INDEX_DIR)}")
print(f"  num_tables    : {qp.config.num_tables}")
print(f"  hash_size     : {qp.config.hash_size}")
print(f"  max_candidates: {qp.config.max_candidates}")
print(f"  Queries run   : {stats['total_queries']}")
print(f"  Mean candidates: {np.mean(candidate_counts):.0f}")
print(f"  Mean latency   : {stats['mean_latency_ms']:.1f} ms")
print("=" * 60)
