"""
scaling_benchmark.py — Milestone 3 Task 5: Scaling Benchmarks
Project 20: Distributed Reverse Image Search Engine
Author: Hayatullah

What this file does:
  1. Builds a 100K sub-sampled index from the existing 1M tables
  2. Builds a synthetic 10M index by perturbing the 1M embeddings (Kaggle only)
  3. BenchmarkRunner — lightweight LSH query executor with P50/P95/P99 stats
  4. Strong scaling experiment — fixed 1M corpus, vary worker count (1→max cores)
  5. Weak scaling experiment — scale corpus + workers together (100K/1, 1M/10, 10M/100)
  6. Saves scaling_results.json and three PNG plots

Run from the repo root on Kaggle:
    python benchmarks/scaling_benchmark.py

Run locally (skips 10M, uses 100K + 1M only):
    python benchmarks/scaling_benchmark.py
"""

import os
import sys
import json
import pickle
import time
import platform
import concurrent.futures

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — works on Kaggle + headless
import matplotlib.pyplot as plt
import h5py

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from lsh_index.lsh_structure import (
    LSHConfig,
    hash_embedding,
    hash_embeddings_batch,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
INDEX_DIR      = os.path.join(_REPO_ROOT, 'lsh_index')
TABLES_DIR     = os.path.join(INDEX_DIR, 'hash_tables')
CNN_PATH       = os.path.join(_REPO_ROOT, 'data', 'all_cnn_embeddings.h5')
GT_PATH        = os.path.join(_REPO_ROOT, 'validation', 'ground_truth.json')
CONFIG_PATH    = os.path.join(INDEX_DIR, 'lsh_config.json')
MATRICES_PATH  = os.path.join(INDEX_DIR, 'projection_matrices.pkl')

BENCH_DIR      = os.path.dirname(os.path.abspath(__file__))
IDX_100K_DIR   = os.path.join(BENCH_DIR, 'indices', '100k')
IDX_10M_DIR    = os.path.join(BENCH_DIR, 'indices', '10m')
RESULTS_PATH   = os.path.join(BENCH_DIR, 'scaling_results.json')


# =============================================================================
# SECTION 1 — Build 100K sub-sampled index
# =============================================================================

def build_100k_index(tables_dir: str, output_dir: str, config: LSHConfig) -> str:
    """
    Sub-sample the existing 1M tables to create a ~100K index.

    Strategy: keep only buckets where hash(bucket_key) % 10 == 0.
    This is deterministic — the same keys are always kept — and gives
    roughly 1/10th of all images (~100K out of 1M).

    Why not just slice 100K IDs?
        Slicing by ID would not preserve bucket structure. Keeping whole buckets
        ensures the index behaves the same way structurally — queries still
        fan out across all 10 tables, just with smaller buckets.

    Args:
        tables_dir : folder containing table_0.pkl … table_9.pkl
        output_dir : where to save table_N_100k.pkl files
        config     : LSHConfig (needs num_tables)

    Returns:
        output_dir path (for confirmation)
    """
    os.makedirs(output_dir, exist_ok=True)
    total_ids_kept = 0

    print(f"\n[100K Index] Sub-sampling {config.num_tables} tables ...")

    for i in range(config.num_tables):
        src_path = os.path.join(tables_dir, f'table_{i}.pkl')
        dst_path = os.path.join(output_dir, f'table_{i}_100k.pkl')

        # Skip if already built
        if os.path.exists(dst_path):
            with open(dst_path, 'rb') as f:
                t = pickle.load(f)
            total_ids_kept += sum(len(v) for v in t.values())
            print(f"  table_{i}_100k.pkl already exists — skipping")
            continue

        with open(src_path, 'rb') as f:
            full_table = pickle.load(f)

        # Keep only buckets whose key hashes to 0 mod 10
        # hash() is deterministic within a Python session for strings
        sub_table = {
            key: ids
            for key, ids in full_table.items()
            if hash(key) % 10 == 0
        }

        ids_kept = sum(len(v) for v in sub_table.values())
        total_ids_kept += ids_kept

        with open(dst_path, 'wb') as f:
            pickle.dump(sub_table, f, protocol=4)

        size_kb = os.path.getsize(dst_path) / 1024
        print(f"  table_{i}_100k.pkl — {len(sub_table):,} buckets, "
              f"{ids_kept:,} IDs, {size_kb:.0f} KB")

    print(f"[100K Index] Done. Total IDs in sub-sampled index: ~{total_ids_kept:,}")
    return output_dir


# =============================================================================
# SECTION 2 — Build 10M synthetic expanded index
# =============================================================================

def build_10m_index(cnn_path: str,
                    tables_dir: str,
                    output_dir: str,
                    projection_matrices: list,
                    config: LSHConfig) -> str:
    """
    Expand the 1M index to 10M by generating 9 synthetic perturbed copies.

    Why synthetic expansion is valid for scaling benchmarks:
        The goal is to measure HOW LATENCY CHANGES as index size grows, not
        to measure accuracy on real images. What matters is that the hash
        tables genuinely contain 10M entries so bucket sizes scale up
        proportionally — giving real query fan-out behaviour at 10M scale.

        Each synthetic copy adds small Gaussian noise (std=0.005) to the
        original embeddings and re-normalizes, preserving the approximate
        cosine similarity structure while creating distinct IDs.

    Memory strategy:
        All 10M embeddings are never held in RAM simultaneously.
        One copy at a time is generated, hashed into tables, and discarded.
        Peak RAM usage: ~512MB (1M embeddings) + tables.

    Args:
        cnn_path            : path to all_cnn_embeddings.h5
        tables_dir          : folder with original table_N.pkl files
        output_dir          : where to save table_N_10m.pkl files
        projection_matrices : list of 10 (12,128) numpy arrays
        config              : LSHConfig

    Returns:
        output_dir path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if already built
    first_table_path = os.path.join(output_dir, 'table_0_10m.pkl')
    if os.path.exists(first_table_path):
        print("[10M Index] Already exists — skipping build.")
        return output_dir

    print("\n[10M Index] Building synthetic 10M index ...")
    print("  This requires ~30–60 minutes and ~20GB disk on Kaggle.")

    # ── Load original 1M embeddings ───────────────────────────────────────
    print("  Loading 1M base embeddings ...")
    with h5py.File(cnn_path, 'r') as f:
        base_embeddings = f['embeddings'][:]   # (1M, 128) float32
        base_ids        = f['image_ids'][:]    # (1M,) int64

    # L2-normalize base embeddings (they should already be normalized)
    norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    base_embeddings = base_embeddings / norms

    # ── Start with copy 0 — the original 1M tables ────────────────────────
    print("  Loading copy 0 (original 1M tables) ...")
    hash_tables_10m = []
    for i in range(config.num_tables):
        src = os.path.join(tables_dir, f'table_{i}.pkl')
        with open(src, 'rb') as f:
            hash_tables_10m.append(pickle.load(f))

    total_ids = sum(sum(len(v) for v in t.values()) for t in hash_tables_10m)
    print(f"  Copy 0 loaded: {total_ids:,} total IDs across all tables")

    # ── Generate 9 synthetic copies ────────────────────────────────────────
    rng = np.random.default_rng(seed=42)

    for copy_idx in range(1, 10):
        start_id = copy_idx * 1_000_000
        end_id   = start_id + 1_000_000

        print(f"\n  Generating copy {copy_idx} (IDs {start_id:,}–{end_id-1:,}) ...")
        t0 = time.time()

        # Add small Gaussian noise — preserves embedding direction
        noise     = rng.normal(loc=0.0, scale=0.005,
                               size=base_embeddings.shape).astype(np.float32)
        perturbed = base_embeddings + noise

        # Re-normalize to unit length
        norms     = np.linalg.norm(perturbed, axis=1, keepdims=True)
        norms     = np.where(norms < 1e-8, 1e-8, norms)
        perturbed = perturbed / norms   # (1M, 128) float32

        # Assign new IDs
        new_ids = np.arange(start_id, end_id, dtype=np.int64)

        # Hash into all 10 tables using vectorized batch function
        for table_idx in range(config.num_tables):
            bucket_keys = hash_embeddings_batch(
                perturbed, projection_matrices[table_idx]
            )
            for img_id, key in zip(new_ids, bucket_keys):
                if key in hash_tables_10m[table_idx]:
                    hash_tables_10m[table_idx][key].append(int(img_id))
                else:
                    hash_tables_10m[table_idx][key] = [int(img_id)]

        elapsed = time.time() - t0
        print(f"  Copy {copy_idx} hashed in {elapsed:.1f}s")

        # Free memory immediately
        del noise, perturbed, new_ids, bucket_keys

    # ── Save all 10 tables ────────────────────────────────────────────────
    print("\n  Saving 10M tables ...")
    for i, table in enumerate(hash_tables_10m):
        dst = os.path.join(output_dir, f'table_{i}_10m.pkl')
        with open(dst, 'wb') as f:
            pickle.dump(table, f, protocol=4)
        size_mb    = os.path.getsize(dst) / (1024 * 1024)
        total_ids  = sum(len(v) for v in table.values())
        print(f"  table_{i}_10m.pkl → {size_mb:.0f} MB  ({total_ids:,} IDs)")

    print("[10M Index] Build complete.")
    return output_dir


# =============================================================================
# SECTION 3 — BenchmarkRunner
# =============================================================================

class BenchmarkRunner:
    """
    Lightweight LSH query executor for benchmarking.

    Does NOT load the full SearchEngine (no feature files, no re-ranking).
    Only runs Phase 1 of search: LSH table fan-out → candidate set.
    This isolates the index lookup latency from re-ranking latency.

    Why not use SearchEngine?
        SearchEngine loads ~1.4GB of feature files at startup.
        For scaling benchmarks where you need to swap index sizes quickly,
        a lightweight runner that only loads the hash tables is much faster
        to initialise and lets you focus specifically on LSH lookup scaling.

    Args:
        tables_dir     : folder containing table_N.pkl (or table_N_100k.pkl etc.)
        table_suffix   : filename suffix, e.g. '' for table_N.pkl,
                         '_100k' for table_N_100k.pkl,
                         '_10m'  for table_N_10m.pkl
        config         : LSHConfig
        projection_matrices : list of 10 projection matrices
    """

    def __init__(self,
                 tables_dir: str,
                 config: LSHConfig,
                 projection_matrices: list,
                 table_suffix: str = ''):
        self.config              = config
        self.projection_matrices = projection_matrices
        self.table_suffix        = table_suffix

        # ── Load hash tables ──────────────────────────────────────────────
        print(f"  Loading tables from {tables_dir} (suffix='{table_suffix}') ...")
        t0 = time.time()
        self.hash_tables = []

        for i in range(config.num_tables):
            fname = f'table_{i}{table_suffix}.pkl'
            path  = os.path.join(tables_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Table not found: {path}")
            with open(path, 'rb') as f:
                self.hash_tables.append(pickle.load(f))

        elapsed     = time.time() - t0
        total_ids   = sum(
            sum(len(v) for v in t.values())
            for t in self.hash_tables
        ) // config.num_tables   # per-table average

        print(f"  Loaded {config.num_tables} tables in {elapsed:.1f}s  "
              f"(~{total_ids:,} IDs per table)")

    def _query_one_table(self, query_embedding: np.ndarray,
                         table_idx: int) -> list:
        """Single table lookup — called in parallel by get_candidates()."""
        key = hash_embedding(query_embedding, self.projection_matrices[table_idx])
        return self.hash_tables[table_idx].get(key, [])

    def get_candidates(self, query_embedding: np.ndarray,
                       num_workers: int) -> set:
        """
        Fan out across all tables using num_workers threads and collect candidates.

        Args:
            query_embedding : normalized (128,) float32
            num_workers     : ThreadPoolExecutor size

        Returns:
            set of candidate image IDs
        """
        per_table = [None] * self.config.num_tables

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            futures = {
                executor.submit(self._query_one_table, query_embedding, i): i
                for i in range(self.config.num_tables)
            }
            for fut in concurrent.futures.as_completed(futures):
                per_table[futures[fut]] = fut.result()

        candidates = set()
        for id_list in per_table:
            candidates.update(id_list)

        # Cap at max_candidates
        if len(candidates) > self.config.max_candidates:
            candidates = set(list(candidates)[:self.config.max_candidates])

        return candidates

    def run_queries(self,
                    query_embeddings: list,
                    num_workers: int) -> dict:
        """
        Run all queries sequentially and collect per-query latency.

        Queries are run ONE AT A TIME even though each query fans out
        across tables using threads. This measures per-query latency,
        not throughput (which would require concurrent callers).

        Args:
            query_embeddings : list of (128,) float32 numpy arrays
            num_workers      : threads used per query for table fan-out

        Returns:
            dict with keys: mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms,
                            num_queries, num_workers
        """
        times_ms = []

        for emb in query_embeddings:
            t0 = time.time()
            _  = self.get_candidates(emb, num_workers)
            elapsed_ms = (time.time() - t0) * 1000
            times_ms.append(elapsed_ms)

        arr = np.array(times_ms)
        return {
            'num_queries': len(times_ms),
            'num_workers': num_workers,
            'mean_ms':     round(float(np.mean(arr)),             2),
            'p50_ms':      round(float(np.percentile(arr, 50)),   2),
            'p95_ms':      round(float(np.percentile(arr, 95)),   2),
            'p99_ms':      round(float(np.percentile(arr, 99)),   2),
            'min_ms':      round(float(np.min(arr)),              2),
            'max_ms':      round(float(np.max(arr)),              2),
        }


# =============================================================================
# SECTION 4 — Load query embeddings from ground truth
# =============================================================================

def load_query_embeddings(gt_path: str,
                          cnn_path: str,
                          n_queries: int = 200) -> list:
    """
    Load CNN embeddings for benchmark queries.

    Tries ground_truth.json IDs first. If those IDs are not in the
    HDF5 file (e.g. they are test-set IDs above 1M), falls back to
    sampling random IDs directly from the HDF5 file.
    """
    with h5py.File(cnn_path, 'r') as f:
        all_ids  = f['image_ids'][:]
        all_embs = f['embeddings'][:]

    id_to_idx = {int(gid): idx for idx, gid in enumerate(all_ids)}

    # Try ground truth IDs first
    try:
        with open(gt_path) as f:
            gt = json.load(f)
        query_ids = [int(k) for k in list(gt.keys())[:n_queries]]
        matched   = [qid for qid in query_ids if qid in id_to_idx]
    except Exception:
        matched = []

    # If fewer than 10 matched, fall back to random IDs from the HDF5
    if len(matched) < 10:
        print(f"  Ground truth IDs not in HDF5 (IDs are test-set images).")
        print(f"  Falling back to {n_queries} random IDs from the 1M index.")
        rng       = np.random.default_rng(seed=42)
        indices   = rng.choice(len(all_ids), size=min(n_queries, len(all_ids)),
                               replace=False)
        matched_idx = indices.tolist()
        embeddings  = []
        for idx in matched_idx:
            emb  = all_embs[idx].astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm
            embeddings.append(emb)
        print(f"  Loaded {len(embeddings)} query embeddings from random sampling")
        return embeddings

    # Use matched ground truth IDs
    embeddings = []
    for qid in matched:
        emb  = all_embs[id_to_idx[qid]].astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        embeddings.append(emb)

    print(f"  Loaded {len(embeddings)} query embeddings from ground truth")
    return embeddings


# =============================================================================
# SECTION 5 — Strong scaling experiment
# =============================================================================

def run_strong_scaling(runner_1m: BenchmarkRunner,
                       query_embeddings: list) -> dict:
    """
    Strong scaling: fixed 1M corpus, vary num_workers.

    Worker counts tested: 1, 2, 4, and up to os.cpu_count().
    For each count, run all query_embeddings and record latency stats.

    What we expect to see:
        Latency decreases as workers increase, but not perfectly linearly.
        The GIL and Python thread overhead cause speedup to taper off.
        This is genuine engineering insight — it shows WHERE the bottleneck is.

    Args:
        runner_1m       : BenchmarkRunner loaded with the 1M index
        query_embeddings: list of normalized query vectors

    Returns:
        dict keyed by num_workers, each value is the stats dict from run_queries()
    """
    max_workers = os.cpu_count() or 4
    # Build worker list: always include 1, 2, 4 and the max available
    worker_counts = sorted(set([1, 2, 4, max_workers]))

    print(f"\n[Strong Scaling] Fixed dataset: 1M images")
    print(f"  Worker counts to test: {worker_counts}")
    print(f"  Queries per config   : {len(query_embeddings)}")

    results = {}

    for nw in worker_counts:
        print(f"\n  Testing {nw} worker(s) ...")
        stats = runner_1m.run_queries(query_embeddings, num_workers=nw)
        results[str(nw)] = stats
        print(f"    P50={stats['p50_ms']:.1f}ms  "
              f"P95={stats['p95_ms']:.1f}ms  "
              f"P99={stats['p99_ms']:.1f}ms")

    return results


# =============================================================================
# SECTION 6 — Weak scaling experiment
# =============================================================================

def run_weak_scaling(query_embeddings: list,
                     config: LSHConfig,
                     projection_matrices: list,
                     idx_100k_dir: str,
                     idx_1m_dir: str,
                     idx_10m_dir: str) -> dict:
    """
    Weak scaling: increase corpus and workers proportionally.

    Three data points:
        (100K images, 1 worker)
        (1M   images, 10 workers)
        (10M  images, 100 workers — capped at os.cpu_count())

    What we expect to see:
        Latency should stay roughly constant if scaling is perfect.
        In practice it increases because larger corpora → larger buckets →
        more candidates per query → more data returned per fan-out call.
        The amount of increase quantifies the system's scaling efficiency.

    Args:
        query_embeddings  : list of normalized query vectors
        config            : LSHConfig
        projection_matrices : list of 10 projection matrices
        idx_100k_dir      : folder with table_N_100k.pkl
        idx_1m_dir        : folder with table_N.pkl (the full 1M tables)
        idx_10m_dir       : folder with table_N_10m.pkl

    Returns:
        dict keyed by corpus size label, each value is the stats dict
    """
    max_workers = os.cpu_count() or 4

    # (corpus label, tables_dir, suffix, num_workers to use)
    configurations = [
        ('100K',  idx_100k_dir, '_100k',  1),
        ('1M',    idx_1m_dir,   '',       min(10, max_workers)),
        ('10M',   idx_10m_dir,  '_10m',   min(100, max_workers)),
    ]

    print(f"\n[Weak Scaling] Proportional corpus + workers")
    print(f"  Queries per config: {len(query_embeddings)}")

    results = {}

    for label, tables_dir, suffix, nw in configurations:
        # Check if index exists
        first_table = os.path.join(tables_dir, f'table_0{suffix}.pkl')
        if not os.path.exists(first_table):
            print(f"\n  [{label}] Index not found at {first_table} — SKIPPING")
            results[label] = {'skipped': True, 'reason': 'index not found'}
            continue

        print(f"\n  [{label}] workers={nw} ...")
        try:
            runner = BenchmarkRunner(
                tables_dir=tables_dir,
                config=config,
                projection_matrices=projection_matrices,
                table_suffix=suffix,
            )
            stats = runner.run_queries(query_embeddings, num_workers=nw)
            stats['corpus_size'] = label
            stats['num_workers'] = nw
            results[label] = stats
            print(f"    P50={stats['p50_ms']:.1f}ms  "
                  f"P95={stats['p95_ms']:.1f}ms  "
                  f"P99={stats['p99_ms']:.1f}ms")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[label] = {'skipped': True, 'reason': str(e)}

    return results


# =============================================================================
# SECTION 7 — Plots
# =============================================================================

def _get_hardware_note() -> str:
    """Return a short hardware description for plot annotations."""
    cpu_count = os.cpu_count() or '?'
    system    = platform.system()
    machine   = platform.machine()
    try:
        # Works on Linux (Kaggle)
        with open('/proc/cpuinfo') as f:
            for line in f:
                if 'model name' in line:
                    cpu_model = line.split(':')[1].strip()
                    return f"{cpu_model} · {cpu_count} cores · {system}"
    except Exception:
        pass
    return f"{system} {machine} · {cpu_count} cores"


def plot_strong_scaling_latency(strong_results: dict, output_path: str):
    """
    Plot P50, P95, P99 latency vs number of workers.

    Shows how adding workers reduces latency and where the benefit
    starts to taper off (due to Python GIL and thread scheduling overhead).
    """
    worker_counts = sorted([int(k) for k in strong_results.keys()])
    if not worker_counts:
        print("  No strong scaling data to plot.")
        return

    p50 = [strong_results[str(nw)]['p50_ms'] for nw in worker_counts]
    p95 = [strong_results[str(nw)]['p95_ms'] for nw in worker_counts]
    p99 = [strong_results[str(nw)]['p99_ms'] for nw in worker_counts]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(worker_counts, p50, 'o-', color='#2196F3', linewidth=2,
            markersize=7, label='P50 (median)')
    ax.plot(worker_counts, p95, 's-', color='#FF9800', linewidth=2,
            markersize=7, label='P95')
    ax.plot(worker_counts, p99, '^-', color='#F44336', linewidth=2,
            markersize=7, label='P99')

    ax.set_xlabel('Number of Workers (threads)', fontsize=12)
    ax.set_ylabel('Query Latency (ms)', fontsize=12)
    ax.set_title('Strong Scaling — LSH Query Latency vs Worker Count\n'
                 '(Fixed corpus: 1M images)', fontsize=13)
    ax.set_xticks(worker_counts)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    hw_note = _get_hardware_note()
    ax.annotate(f'Hardware: {hw_note}',
                xy=(0.01, 0.02), xycoords='axes fraction',
                fontsize=8, color='grey')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_strong_scaling_speedup(strong_results: dict, output_path: str):
    """
    Plot speedup ratio vs number of workers.

    speedup(N) = latency(1 worker) / latency(N workers)

    Compares actual speedup against ideal linear speedup.
    The gap between the two lines reveals parallelism inefficiency
    caused by Python GIL contention and thread scheduling overhead.
    """
    worker_counts = sorted([int(k) for k in strong_results.keys()])
    if not worker_counts or '1' not in strong_results:
        print("  No strong scaling data to plot speedup (need 1-worker baseline).")
        return

    baseline_p50 = strong_results['1']['p50_ms']
    baseline_p99 = strong_results['1']['p99_ms']

    actual_p50  = [baseline_p50 / strong_results[str(nw)]['p50_ms']
                   for nw in worker_counts]
    actual_p99  = [baseline_p99 / strong_results[str(nw)]['p99_ms']
                   for nw in worker_counts]
    ideal       = [float(nw) for nw in worker_counts]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(worker_counts, ideal,      '--', color='#9E9E9E', linewidth=1.5,
            label='Ideal linear speedup')
    ax.plot(worker_counts, actual_p50, 'o-', color='#2196F3', linewidth=2,
            markersize=7, label='Actual speedup (P50)')
    ax.plot(worker_counts, actual_p99, '^-', color='#F44336', linewidth=2,
            markersize=7, label='Actual speedup (P99)')

    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Speedup (relative to 1 worker)', fontsize=12)
    ax.set_title('Strong Scaling — Speedup Ratio vs Worker Count\n'
                 '(Fixed corpus: 1M images)', fontsize=13)
    ax.set_xticks(worker_counts)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    hw_note = _get_hardware_note()
    ax.annotate(f'Hardware: {hw_note}',
                xy=(0.01, 0.02), xycoords='axes fraction',
                fontsize=8, color='grey')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_weak_scaling_latency(weak_results: dict, output_path: str):
    """
    Plot P50, P95, P99 latency vs corpus size (log scale).

    In a perfectly scalable system, all three lines would be flat
    (latency constant as corpus and workers both scale up).
    The upward slope reveals scaling inefficiency — primarily caused
    by larger bucket sizes at 10M producing more re-ranking candidates.
    """
    # Filter out skipped configurations
    labels_ordered = ['100K', '1M', '10M']
    valid = [(l, weak_results[l]) for l in labels_ordered
             if l in weak_results and not weak_results[l].get('skipped')]

    if not valid:
        print("  No weak scaling data to plot.")
        return

    labels = [v[0] for v in valid]
    p50    = [v[1]['p50_ms'] for v in valid]
    p95    = [v[1]['p95_ms'] for v in valid]
    p99    = [v[1]['p99_ms'] for v in valid]
    workers = [str(v[1]['num_workers']) for v in valid]

    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, p50, 'o-', color='#2196F3', linewidth=2,
            markersize=8, label='P50 (median)')
    ax.plot(x, p95, 's-', color='#FF9800', linewidth=2,
            markersize=8, label='P95')
    ax.plot(x, p99, '^-', color='#F44336', linewidth=2,
            markersize=8, label='P99')

    # Annotate each point with its worker count
    for i, (lbl, nw) in enumerate(zip(labels, workers)):
        ax.annotate(f'{nw}w', xy=(i, p99[i]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, color='#F44336')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n({w} workers)' for l, w in zip(labels, workers)],
                       fontsize=11)
    ax.set_ylabel('Query Latency (ms)', fontsize=12)
    ax.set_title('Weak Scaling — Latency vs Corpus Size\n'
                 '(Corpus and workers scaled proportionally)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation explaining ideal behaviour
    ax.axhline(y=p50[0], color='#2196F3', linestyle=':', alpha=0.5)
    ax.annotate('Ideal: flat line (P50)',
                xy=(len(labels)-1, p50[0]),
                xytext=(-10, 8), textcoords='offset points',
                fontsize=8, color='#2196F3', alpha=0.7)

    hw_note = _get_hardware_note()
    ax.annotate(f'Hardware: {hw_note}',
                xy=(0.01, 0.02), xycoords='axes fraction',
                fontsize=8, color='grey')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# SECTION 8 — Main entry point
# =============================================================================

if __name__ == '__main__':

    print("=" * 65)
    print("Milestone 3 Task 5 — Scaling Benchmarks")
    print("=" * 65)

    # ── Check prerequisites ───────────────────────────────────────────────
    for path in [CONFIG_PATH, MATRICES_PATH]:
        if not os.path.exists(path):
            print(f"\nERROR: {path} not found.")
            print("Run lsh_structure.py first.")
            sys.exit(1)

    if not os.path.exists(CNN_PATH):
        print(f"\nERROR: {CNN_PATH} not found.")
        print("Attach the MIRFLICKR-1M features dataset.")
        sys.exit(1)

    if not os.path.exists(GT_PATH):
        print(f"\nERROR: {GT_PATH} not found.")
        sys.exit(1)

    # ── Load config and matrices ──────────────────────────────────────────
    print("\n[Setup] Loading config and projection matrices ...")
    with open(CONFIG_PATH) as f:
        config = LSHConfig.from_dict(json.load(f))
    with open(MATRICES_PATH, 'rb') as f:
        projection_matrices = pickle.load(f)

    print(f"  Config: {config.num_tables} tables, hash_size={config.hash_size}")

    # ── Step 1: Build 100K index ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 1 — Building 100K sub-sampled index")
    print("=" * 65)
    build_100k_index(TABLES_DIR, IDX_100K_DIR, config)

    # ── Step 2: Build 10M index (Kaggle only, skipped locally) ───────────
    print("\n" + "=" * 65)
    print("STEP 2 — Building 10M synthetic index")
    print("=" * 65)

    if not os.path.exists(IDX_10M_DIR):
        os.makedirs(IDX_10M_DIR, exist_ok=True)

    first_10m = os.path.join(IDX_10M_DIR, 'table_0_10m.pkl')
    if not os.path.exists(first_10m):
        # Check available disk space — need ~20GB
        try:
            import shutil
            free_gb = shutil.disk_usage(IDX_10M_DIR).free / (1024 ** 3)
            if free_gb < 20:
                print(f"  WARNING: Only {free_gb:.1f}GB free disk space.")
                print(f"  10M index requires ~20GB. Skipping 10M build.")
                print(f"  Run on Kaggle (100GB disk) where space is available.")
            else:
                build_10m_index(CNN_PATH, TABLES_DIR, IDX_10M_DIR,
                                projection_matrices, config)
        except Exception as e:
            print(f"  Could not build 10M index: {e}")
            print(f"  Skipping — benchmark will run 100K and 1M only.")
    else:
        print(f"  10M index already exists at {IDX_10M_DIR}")

    # ── Load query embeddings ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — Loading query embeddings")
    print("=" * 65)
    query_embeddings = load_query_embeddings(GT_PATH, CNN_PATH, n_queries=200)

    if len(query_embeddings) < 10:
        print("ERROR: Need at least 10 query embeddings. Check ground_truth.json.")
        sys.exit(1)

    # ── Strong scaling ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4 — Strong Scaling (fixed 1M corpus, vary workers)")
    print("=" * 65)

    runner_1m = BenchmarkRunner(
        tables_dir=TABLES_DIR,
        config=config,
        projection_matrices=projection_matrices,
        table_suffix='',
    )
    strong_results = run_strong_scaling(runner_1m, query_embeddings)

    # ── Weak scaling ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 5 — Weak Scaling (proportional corpus + workers)")
    print("=" * 65)

    weak_results = run_weak_scaling(
        query_embeddings    = query_embeddings,
        config              = config,
        projection_matrices = projection_matrices,
        idx_100k_dir        = IDX_100K_DIR,
        idx_1m_dir          = TABLES_DIR,
        idx_10m_dir         = IDX_10M_DIR,
    )

    # ── Save results JSON ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 6 — Saving results and plots")
    print("=" * 65)

    results = {
        'hardware':      _get_hardware_note(),
        'n_queries':     len(query_embeddings),
        'strong_scaling': strong_results,
        'weak_scaling':   weak_results,
    }

    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {RESULTS_PATH}")

    # ── Generate plots ────────────────────────────────────────────────────
    print("\n  Generating plots ...")

    plot_strong_scaling_latency(
        strong_results,
        os.path.join(BENCH_DIR, 'strong_scaling_latency.png')
    )
    plot_strong_scaling_speedup(
        strong_results,
        os.path.join(BENCH_DIR, 'strong_scaling_speedup.png')
    )
    plot_weak_scaling_latency(
        weak_results,
        os.path.join(BENCH_DIR, 'weak_scaling_latency.png')
    )

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    print("\nStrong Scaling (1M corpus):")
    print(f"  {'Workers':<10} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}")
    print("  " + "-" * 40)
    for nw in sorted([int(k) for k in strong_results.keys()]):
        s = strong_results[str(nw)]
        print(f"  {nw:<10} {s['p50_ms']:>10.1f} {s['p95_ms']:>10.1f} "
              f"{s['p99_ms']:>10.1f}")

    print("\nWeak Scaling:")
    print(f"  {'Corpus':<8} {'Workers':>8} {'P50 (ms)':>10} "
          f"{'P95 (ms)':>10} {'P99 (ms)':>10}")
    print("  " + "-" * 50)
    for label in ['100K', '1M', '10M']:
        if label not in weak_results:
            continue
        s = weak_results[label]
        if s.get('skipped'):
            print(f"  {label:<8} {'—':>8} {'SKIPPED':>10}")
        else:
            print(f"  {label:<8} {s['num_workers']:>8} "
                  f"{s['p50_ms']:>10.1f} {s['p95_ms']:>10.1f} "
                  f"{s['p99_ms']:>10.1f}")

    print("\n" + "=" * 65)
    print("Task 5 COMPLETE. Files produced:")
    print(f"  {RESULTS_PATH}")
    print(f"  {os.path.join(BENCH_DIR, 'strong_scaling_latency.png')}")
    print(f"  {os.path.join(BENCH_DIR, 'strong_scaling_speedup.png')}")
    print(f"  {os.path.join(BENCH_DIR, 'weak_scaling_latency.png')}")
    print(f"  {IDX_100K_DIR}/table_N_100k.pkl  (10 files)")
    print(f"  {IDX_10M_DIR}/table_N_10m.pkl    (Kaggle only)")
    print("=" * 65)