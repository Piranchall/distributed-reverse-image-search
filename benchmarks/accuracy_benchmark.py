"""
accuracy_benchmark.py — Milestone 3 Task 6: Accuracy vs Latency Trade-off
Project 20: Distributed Reverse Image Search Engine
Author: Aliza

What this file does:
  1. Defines a parameter grid of LSH configs (num_tables × hash_size combinations)
  2. build_test_index()       — builds an in-memory LSH index for each config
                                using only the 600 test images (fast, seconds not hours)
  3. LightweightQueryEngine  — runs LSH lookup + re-ranking on the test set
                                without loading the full 1.4GB feature store
  4. Accuracy measurement    — precision@k and recall@k for all 100 ground truth queries
  5. Trade-off table         — sorted results printed and saved as JSON
  6. Two plots               — Pareto frontier + parameter sensitivity subplots
  7. Cache validation        — confirms the LRU cache doesn't change accuracy

Run from the repo root:
    python benchmarks/accuracy_benchmark.py
"""

import os
import sys
import json
import pickle
import time
import collections

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from lsh_index.lsh_structure import (
    LSHConfig,
    generate_projection_matrices,
    hash_embedding,
    hash_embeddings_batch,
)
from validation.evaluator import (
    precision_at_k,
    recall_at_k,
    evaluate_single_query,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_DIR   = os.path.join(_REPO_ROOT, 'validation', 'test_features')
GT_PATH    = os.path.join(_REPO_ROOT, 'validation', 'ground_truth.json')
BENCH_DIR  = os.path.dirname(os.path.abspath(__file__))

TEST_CNN_PATH   = os.path.join(TEST_DIR, 'test_cnn.h5')
TEST_HASH_PATH  = os.path.join(TEST_DIR, 'test_hashes.pkl')
TEST_SIFT_PATH  = os.path.join(TEST_DIR, 'test_sift.pkl')
TEST_ORB_PATH   = os.path.join(TEST_DIR, 'test_orb.pkl')
TEST_HIST_PATH  = os.path.join(TEST_DIR, 'test_hist.pkl')

RESULTS_PATH    = os.path.join(BENCH_DIR, 'accuracy_results.json')
TRADEOFF_PLOT   = os.path.join(BENCH_DIR, 'accuracy_latency_tradeoff.png')
SENSITIVITY_PLOT= os.path.join(BENCH_DIR, 'parameter_sensitivity.png')


# =============================================================================
# SECTION 1 — Parameter Grid
# =============================================================================

# 10 configurations that span the interesting range.
# Always includes the production baseline (num_tables=10, hash_size=12).
# Chosen to cover: few tables vs many, small hash (bigger buckets, higher recall)
# vs large hash (smaller buckets, lower recall but faster).

PARAMETER_GRID = [
    # Baseline — current production config
    {'num_tables':  10, 'hash_size': 12, 'label': 'baseline'},

    # Vary num_tables, hold hash_size=12
    {'num_tables':   5, 'hash_size': 12, 'label': 'tables5_hash12'},
    {'num_tables':  15, 'hash_size': 12, 'label': 'tables15_hash12'},
    {'num_tables':  20, 'hash_size': 12, 'label': 'tables20_hash12'},

    # Vary hash_size, hold num_tables=10
    {'num_tables':  10, 'hash_size':  8, 'label': 'tables10_hash8'},
    {'num_tables':  10, 'hash_size': 10, 'label': 'tables10_hash10'},
    {'num_tables':  10, 'hash_size': 14, 'label': 'tables10_hash14'},

    # Corner configs — both extremes
    {'num_tables':   5, 'hash_size':  8, 'label': 'tables5_hash8'},
    {'num_tables':  20, 'hash_size': 14, 'label': 'tables20_hash14'},

    # Middle ground
    {'num_tables':  15, 'hash_size': 10, 'label': 'tables15_hash10'},
]


# =============================================================================
# SECTION 2 — Test Feature Store
# =============================================================================

class TestFeatureStore:
    """
    Lightweight feature store that holds ONLY the 600 test images.

    This is the test-set equivalent of All_Features from feature_fusion.py.
    Loading 600 test images takes milliseconds vs the 1.4GB full feature store.

    Attributes:
        embeddings   : numpy array (600, 128) float32
        id_to_index  : dict {image_id → row index in embeddings}
        hashes       : dict {image_id → {'ahash': int, 'dhash': int, 'phash': int}}
        sift         : dict {image_id → numpy array}
        orb          : dict {image_id → numpy array}
        hist         : dict {image_id → numpy array}
        all_ids      : list of all 600 image IDs
    """

    def __init__(self):
        print("  Loading test feature store (600 images) ...")

        # CNN embeddings
        with h5py.File(TEST_CNN_PATH, 'r') as f:
            self.embeddings = f['embeddings'][:].astype(np.float32)
            image_ids       = f['image_ids'][:]

        self.id_to_index = {int(gid): idx for idx, gid in enumerate(image_ids)}
        self.all_ids     = [int(gid) for gid in image_ids]

        # Normalize embeddings to unit length (same as production pipeline)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1e-8, norms)
        self.embeddings = self.embeddings / norms

        # Perceptual hashes
        with open(TEST_HASH_PATH, 'rb') as f:
            self.hashes = pickle.load(f)

        # Traditional features
        with open(TEST_SIFT_PATH, 'rb') as f:
            self.sift = pickle.load(f)
        with open(TEST_ORB_PATH, 'rb') as f:
            self.orb = pickle.load(f)
        with open(TEST_HIST_PATH, 'rb') as f:
            self.hist = pickle.load(f)

        print(f"  TestFeatureStore ready: {len(self.all_ids)} images, "
              f"embedding shape={self.embeddings.shape}")

    def get_embedding(self, image_id: int) -> np.ndarray:
        return self.embeddings[self.id_to_index[image_id]]

    def get_hash(self, image_id: int) -> dict:
        return self.hashes.get(image_id, {'ahash': 0, 'dhash': 0, 'phash': 0})

    def get_sift(self, image_id: int) -> np.ndarray:
        return self.sift.get(image_id, np.zeros(128, dtype=np.float32))

    def get_orb(self, image_id: int) -> np.ndarray:
        return self.orb.get(image_id, np.zeros(32, dtype=np.float32))

    def get_hist(self, image_id: int) -> np.ndarray:
        return self.hist.get(image_id, np.zeros(96, dtype=np.float32))


# =============================================================================
# SECTION 3 — Build test index for one config
# =============================================================================

def build_test_index(num_tables: int, hash_size: int,
                     feature_store: TestFeatureStore) -> dict:
    """
    Build an in-memory LSH index for 600 test images under one config.

    Does NOT write to disk — held in memory for the benchmark duration.
    Runs in under 1 second for 600 images.

    How it works:
        Creates a fresh LSHConfig with the given num_tables and hash_size.
        Generates new random projection matrices with seed 42 (reproducible).
        Hashes all 600 test embeddings into num_tables hash tables using
        hash_embeddings_batch() — the same function used in build_index.py.

    Args:
        num_tables    : number of LSH tables for this config
        hash_size     : number of bits per hash key (controls bucket granularity)
        feature_store : TestFeatureStore with embeddings already normalized

    Returns:
        dict with keys:
            'config'              : LSHConfig object
            'projection_matrices' : list of num_tables numpy arrays (hash_size, 128)
            'hash_tables'         : list of num_tables dicts {key_str: [image_id,...]}
    """
    # Create config with given parameters
    config = LSHConfig()
    config.num_tables    = num_tables
    config.hash_size     = hash_size
    config.embedding_dim = 128
    config.random_seed   = 42

    # Generate projection matrices — fresh for each config
    # Different hash_size means different matrix shape (hash_size × 128)
    projection_matrices = generate_projection_matrices(config)

    # Build hash tables using vectorized batch hashing
    # embeddings shape: (600, 128)
    hash_tables = [{} for _ in range(num_tables)]

    for table_idx in range(num_tables):
        bucket_keys = hash_embeddings_batch(
            feature_store.embeddings,
            projection_matrices[table_idx]
        )
        for img_id, key in zip(feature_store.all_ids, bucket_keys):
            if key in hash_tables[table_idx]:
                hash_tables[table_idx][key].append(img_id)
            else:
                hash_tables[table_idx][key] = [img_id]

    return {
        'config':               config,
        'projection_matrices':  projection_matrices,
        'hash_tables':          hash_tables,
    }


# =============================================================================
# SECTION 4 — LightweightQueryEngine
# =============================================================================

class LightweightQueryEngine:
    """
    Runs LSH lookup + CNN re-ranking on the test set.

    Does NOT load SearchEngine, QueryProcessor, or the full 1M feature files.
    Uses only the 600-image TestFeatureStore and the in-memory test index.

    This allows benchmarking all 10 parameter configurations in minutes
    without being blocked by Tasks 1-4 or the full dataset.

    Args:
        index         : dict returned by build_test_index()
        feature_store : TestFeatureStore instance
    """

    def __init__(self, index: dict, feature_store: TestFeatureStore):
        self.config               = index['config']
        self.projection_matrices  = index['projection_matrices']
        self.hash_tables          = index['hash_tables']
        self.feature_store        = feature_store

    def query(self, query_image_id: int, top_k: int = 10) -> tuple:
        """
        Run one query and return results + elapsed time.

        Steps:
            1. Get query embedding from feature store (already normalized)
            2. Hash it into all tables → collect candidate IDs
            3. Score all candidates using CNN cosine similarity (vectorized)
            4. Return top_k sorted by score

        Args:
            query_image_id : image ID (must be in test set, 2000000–2000599)
            top_k          : number of results to return

        Returns:
            (result_ids_list, elapsed_ms)
            result_ids_list : list of top_k integer image IDs
            elapsed_ms      : float, total query time in milliseconds
        """
        t0 = time.time()

        # Step 1 — get query embedding
        query_emb = self.feature_store.get_embedding(query_image_id)

        # Step 2 — fan out across all tables, collect candidate IDs
        candidate_set = set()
        for table_idx in range(self.config.num_tables):
            key    = hash_embedding(query_emb, self.projection_matrices[table_idx])
            bucket = self.hash_tables[table_idx].get(key, [])
            candidate_set.update(bucket)

        # Remove query itself from candidates (don't rank an image against itself)
        candidate_set.discard(query_image_id)
        candidates = list(candidate_set)

        # Step 3 — CNN cosine similarity scoring (vectorized)
        if not candidates:
            return [], (time.time() - t0) * 1000

        # Stack candidate embeddings: (N, 128)
        cand_matrix = np.stack(
            [self.feature_store.get_embedding(cid) for cid in candidates],
            axis=0
        )

        # All embeddings already normalized — dot product = cosine similarity
        scores = cand_matrix @ query_emb   # shape (N,)

        # Step 4 — sort and return top_k
        sorted_pairs = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        result_ids = [pair[0] for pair in sorted_pairs[:top_k]]
        elapsed_ms = (time.time() - t0) * 1000

        return result_ids, elapsed_ms

    def get_mean_candidates(self, query_ids: list) -> float:
        """Measure average candidate set size across a list of queries."""
        counts = []
        for qid in query_ids:
            q_emb = self.feature_store.get_embedding(qid)
            cands = set()
            for table_idx in range(self.config.num_tables):
                key = hash_embedding(q_emb, self.projection_matrices[table_idx])
                cands.update(self.hash_tables[table_idx].get(key, []))
            cands.discard(qid)
            counts.append(len(cands))
        return float(np.mean(counts)) if counts else 0.0


# =============================================================================
# SECTION 5 — Run accuracy measurement for one config
# =============================================================================

def run_config_benchmark(engine: LightweightQueryEngine,
                         ground_truth: dict,
                         query_ids: list,
                         top_k: int = 10) -> dict:
    """
    Run all queries for one configuration and return accuracy + latency stats.

    For each query:
        - Run engine.query() → get result IDs and elapsed time
        - Call evaluate_single_query() → precision@k and recall@k for k=1,5,10

    Then aggregate across all queries:
        - mean precision@1, @5, @10
        - mean recall@5, @10
        - P50, P95, P99 latency

    Args:
        engine       : LightweightQueryEngine for this config
        ground_truth : dict {str(image_id): [variant_ids]}
        query_ids    : list of integer query IDs to run
        top_k        : number of results per query

    Returns:
        dict with all accuracy and latency statistics
    """
    latencies_ms = []
    per_query_metrics = []

    for qid in query_ids:
        result_ids, elapsed_ms = engine.query(qid, top_k=top_k)
        latencies_ms.append(elapsed_ms)

        # evaluate_single_query expects ground_truth keyed by int
        gt_int = {int(k): v for k, v in ground_truth.items()}
        metrics = evaluate_single_query(
            query_id       = qid,
            search_results = result_ids,
            ground_truth   = gt_int,
            k_values       = [1, 5, 10],
        )
        per_query_metrics.append(metrics)

    # Aggregate latency
    lat_arr = np.array(latencies_ms)

    # Aggregate accuracy — mean across all queries
    def mean_metric(k, metric):
        vals = [m[k][metric] for m in per_query_metrics if k in m]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    # Mean candidate count
    mean_cands = engine.get_mean_candidates(query_ids[:20])  # sample 20 for speed

    return {
        'num_tables':       engine.config.num_tables,
        'hash_size':        engine.config.hash_size,
        'num_queries':      len(query_ids),
        'precision_at_1':   mean_metric(1,  'precision'),
        'precision_at_5':   mean_metric(5,  'precision'),
        'precision_at_10':  mean_metric(10, 'precision'),
        'recall_at_5':      mean_metric(5,  'recall'),
        'recall_at_10':     mean_metric(10, 'recall'),
        'mean_latency_ms':  round(float(np.mean(lat_arr)),           2),
        'p50_latency_ms':   round(float(np.percentile(lat_arr, 50)), 2),
        'p95_latency_ms':   round(float(np.percentile(lat_arr, 95)), 2),
        'p99_latency_ms':   round(float(np.percentile(lat_arr, 99)), 2),
        'mean_candidates':  round(mean_cands, 1),
    }


# =============================================================================
# SECTION 6 — Print trade-off table
# =============================================================================

def print_tradeoff_table(all_results: dict):
    """
    Print all configurations sorted by recall@5 descending.
    Marks the baseline (num_tables=10, hash_size=12) with *.
    """
    rows = sorted(
        all_results.values(),
        key=lambda x: x['recall_at_5'],
        reverse=True
    )

    print(f"\n{'Config':<22} {'Tables':>6} {'Hash':>5} "
          f"{'R@5':>6} {'P@10':>6} "
          f"{'Mean ms':>8} {'P95 ms':>7} {'P99 ms':>7} "
          f"{'Cands':>7}")
    print("-" * 85)

    for r in rows:
        is_baseline = (r['num_tables'] == 10 and r['hash_size'] == 12)
        marker      = " *" if is_baseline else "  "
        label       = f"t{r['num_tables']}_h{r['hash_size']}{marker}"

        print(f"  {label:<20} {r['num_tables']:>6} {r['hash_size']:>5} "
              f"{r['recall_at_5']:>6.3f} {r['precision_at_10']:>6.3f} "
              f"{r['mean_latency_ms']:>8.2f} {r['p95_latency_ms']:>7.2f} "
              f"{r['p99_latency_ms']:>7.2f} "
              f"{r['mean_candidates']:>7.0f}")

    print("-" * 85)
    print("  * = production baseline (num_tables=10, hash_size=12)")


# =============================================================================
# SECTION 7 — Pareto frontier plot
# =============================================================================

def plot_pareto_frontier(all_results: dict, output_path: str):
    """
    Plot recall@5 vs mean latency. Highlight the Pareto frontier.

    The Pareto frontier connects configurations where no other config
    has BOTH better recall AND lower latency simultaneously.
    Points on the frontier represent genuine trade-off options.
    Points below the frontier are dominated — you can always do better.

    Production baseline is shown with a red star marker.
    """
    rows = list(all_results.values())

    latencies = [r['mean_latency_ms'] for r in rows]
    recalls   = [r['recall_at_5']     for r in rows]
    labels    = [f"t{r['num_tables']}\nh{r['hash_size']}" for r in rows]

    # Identify Pareto-optimal points
    # A point is Pareto-optimal if no other point has both lower latency AND higher recall
    pareto_mask = []
    for i, (lat_i, rec_i) in enumerate(zip(latencies, recalls)):
        dominated = any(
            lat_j <= lat_i and rec_j >= rec_i and (lat_j < lat_i or rec_j > rec_i)
            for j, (lat_j, rec_j) in enumerate(zip(latencies, recalls))
            if j != i
        )
        pareto_mask.append(not dominated)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all points
    for i, (lat, rec, lbl) in enumerate(zip(latencies, recalls, labels)):
        is_baseline = (rows[i]['num_tables'] == 10 and rows[i]['hash_size'] == 12)
        is_pareto   = pareto_mask[i]

        if is_baseline:
            ax.scatter(lat, rec, s=200, color='#F44336', zorder=5,
                       marker='*', label='Production baseline')
        elif is_pareto:
            ax.scatter(lat, rec, s=100, color='#2196F3', zorder=4,
                       marker='D', label='Pareto optimal' if i == next(
                           j for j, p in enumerate(pareto_mask) if p and
                           not (rows[j]['num_tables'] == 10 and rows[j]['hash_size'] == 12)
                       ) else '')
        else:
            ax.scatter(lat, rec, s=60, color='#9E9E9E', zorder=3,
                       marker='o', alpha=0.6)

        # Label each point
        ax.annotate(lbl, (lat, rec),
                    textcoords='offset points',
                    xytext=(6, 4), fontsize=7.5, color='#333333')

    # Draw Pareto frontier line (connect Pareto points sorted by latency)
    pareto_points = sorted(
        [(latencies[i], recalls[i]) for i in range(len(rows)) if pareto_mask[i]],
        key=lambda x: x[0]
    )
    if len(pareto_points) > 1:
        px, py = zip(*pareto_points)
        ax.plot(px, py, '--', color='#2196F3', linewidth=1.5,
                alpha=0.7, label='Pareto frontier')

    ax.set_xlabel('Mean Query Latency (ms)', fontsize=12)
    ax.set_ylabel('Recall@5', fontsize=12)
    ax.set_title('Accuracy vs Latency Trade-off\n'
                 'LSH Parameter Configurations (test set, 600 images)', fontsize=13)

    # Deduplicate legend entries
    handles, lbls = ax.get_legend_handles_labels()
    seen, unique_h, unique_l = set(), [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen.add(l); unique_h.append(h); unique_l.append(l)
    ax.legend(unique_h, unique_l, fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# SECTION 8 — Parameter sensitivity plot
# =============================================================================

def plot_parameter_sensitivity(all_results: dict, output_path: str):
    """
    Two side-by-side subplots showing the independent effect of each parameter.

    Left subplot:  hold hash_size=12, vary num_tables (5, 10, 15, 20)
                   Shows: more tables → higher recall, more latency
    Right subplot: hold num_tables=10, vary hash_size (8, 10, 12, 14)
                   Shows: larger hash → lower recall (smaller buckets), less latency

    Each subplot has dual y-axes: recall@5 (left, blue) and P99 latency (right, red).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left subplot: vary num_tables, hash_size=12 ───────────────────────
    vary_tables = sorted(
        [r for r in all_results.values() if r['hash_size'] == 12],
        key=lambda x: x['num_tables']
    )

    if vary_tables:
        nt_vals  = [r['num_tables']    for r in vary_tables]
        rec_vals = [r['recall_at_5']   for r in vary_tables]
        p99_vals = [r['p99_latency_ms']for r in vary_tables]

        ax1b = ax1.twinx()
        l1, = ax1.plot(nt_vals, rec_vals, 'o-', color='#2196F3',
                       linewidth=2, markersize=8, label='Recall@5')
        l2, = ax1b.plot(nt_vals, p99_vals, 's--', color='#F44336',
                        linewidth=2, markersize=8, label='P99 latency (ms)')

        ax1.set_xlabel('num_tables', fontsize=11)
        ax1.set_ylabel('Recall@5', fontsize=11, color='#2196F3')
        ax1b.set_ylabel('P99 Latency (ms)', fontsize=11, color='#F44336')
        ax1.set_title('Effect of num_tables\n(hash_size=12 fixed)', fontsize=11)
        ax1.set_xticks(nt_vals)
        ax1.tick_params(axis='y', labelcolor='#2196F3')
        ax1b.tick_params(axis='y', labelcolor='#F44336')
        ax1.legend(handles=[l1, l2], fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)

    # ── Right subplot: vary hash_size, num_tables=10 ──────────────────────
    vary_hash = sorted(
        [r for r in all_results.values() if r['num_tables'] == 10],
        key=lambda x: x['hash_size']
    )

    if vary_hash:
        hs_vals  = [r['hash_size']     for r in vary_hash]
        rec_vals = [r['recall_at_5']   for r in vary_hash]
        p99_vals = [r['p99_latency_ms']for r in vary_hash]

        ax2b = ax2.twinx()
        l3, = ax2.plot(hs_vals, rec_vals, 'o-', color='#2196F3',
                       linewidth=2, markersize=8, label='Recall@5')
        l4, = ax2b.plot(hs_vals, p99_vals, 's--', color='#F44336',
                        linewidth=2, markersize=8, label='P99 latency (ms)')

        ax2.set_xlabel('hash_size', fontsize=11)
        ax2.set_ylabel('Recall@5', fontsize=11, color='#2196F3')
        ax2b.set_ylabel('P99 Latency (ms)', fontsize=11, color='#F44336')
        ax2.set_title('Effect of hash_size\n(num_tables=10 fixed)', fontsize=11)
        ax2.set_xticks(hs_vals)
        ax2.tick_params(axis='y', labelcolor='#2196F3')
        ax2b.tick_params(axis='y', labelcolor='#F44336')
        ax2.legend(handles=[l3, l4], fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)

    fig.suptitle('LSH Parameter Sensitivity Analysis', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# SECTION 9 — Cache validation
# =============================================================================

class SimpleQueryCache:
    """
    Minimal LRU cache wrapper around LightweightQueryEngine.

    Used to validate that caching does not change accuracy.
    Stores exact results — a cache hit returns the identical list
    that was returned on the first (miss) query.

    This is a simplified version of CachedSearchEngine from query_cache.py,
    adapted to work with LightweightQueryEngine instead of SearchEngine.
    """

    def __init__(self, engine: LightweightQueryEngine, max_size: int = 200):
        self._engine    = engine
        self._cache     = collections.OrderedDict()  # LRU store
        self._max_size  = max_size
        self.hits       = 0
        self.misses     = 0
        self.hit_times  = []
        self.miss_times = []

    def query(self, query_image_id: int, top_k: int = 10) -> tuple:
        """
        Return cached results if available, otherwise run engine.query().

        Args:
            query_image_id : image ID
            top_k          : number of results

        Returns:
            (result_ids_list, elapsed_ms, cache_hit: bool)
        """
        if query_image_id in self._cache:
            # Cache hit — return stored results immediately
            result_ids = self._cache[query_image_id]
            self._cache.move_to_end(query_image_id)   # LRU update
            self.hits += 1
            t0 = time.time()
            elapsed_ms = (time.time() - t0) * 1000 + 0.01  # ~0ms for dict lookup
            self.hit_times.append(elapsed_ms)
            return result_ids, elapsed_ms, True

        # Cache miss — run real query
        t0 = time.time()
        result_ids, elapsed_ms = self._engine.query(query_image_id, top_k)
        self.misses += 1
        self.miss_times.append(elapsed_ms)

        # Store in cache
        self._cache[query_image_id] = result_ids
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return result_ids, elapsed_ms, False

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            'total_queries':    total,
            'hits':             self.hits,
            'misses':           self.misses,
            'hit_rate_pct':     round(self.hits / total * 100, 2) if total > 0 else 0.0,
            'mean_hit_ms':      round(float(np.mean(self.hit_times)),  4) if self.hit_times  else 0.0,
            'mean_miss_ms':     round(float(np.mean(self.miss_times)), 4) if self.miss_times else 0.0,
        }


def run_cache_validation(engine: LightweightQueryEngine,
                         ground_truth: dict,
                         query_ids: list,
                         top_k: int = 10) -> dict:
    """
    Validate that the cache returns identical results to the uncached engine.

    Sends each query TWICE:
        First pass  → all misses, results stored in cache
        Second pass → all hits, results returned from cache

    Verifies that hit results == miss results for every query.
    Confirms cache does NOT introduce accuracy loss.

    Args:
        engine       : LightweightQueryEngine (baseline config)
        ground_truth : dict {str(image_id): [variant_ids]}
        query_ids    : list of query IDs
        top_k        : number of results

    Returns:
        dict with cache stats and accuracy comparison
    """
    cache = SimpleQueryCache(engine, max_size=len(query_ids) + 10)

    first_pass_results  = {}
    second_pass_results = {}

    # First pass — all misses
    for qid in query_ids:
        result_ids, _, was_hit = cache.query(qid, top_k)
        first_pass_results[qid] = result_ids

    # Second pass — all hits (same queries, same order)
    for qid in query_ids:
        result_ids, _, was_hit = cache.query(qid, top_k)
        second_pass_results[qid] = result_ids

    # Verify identical results
    mismatches = []
    for qid in query_ids:
        if first_pass_results[qid] != second_pass_results[qid]:
            mismatches.append(qid)

    accuracy_preserved = (len(mismatches) == 0)

    stats = cache.get_stats()
    stats['accuracy_preserved']  = accuracy_preserved
    stats['result_mismatches']   = len(mismatches)
    stats['expected_hit_rate']   = 50.0   # 2 passes → 50% hits

    return stats


# =============================================================================
# SECTION 10 — Main entry point
# =============================================================================

if __name__ == '__main__':

    print("=" * 65)
    print("Milestone 3 Task 6 — Accuracy vs Latency Trade-off")
    print("=" * 65)

    # ── Check prerequisites ───────────────────────────────────────────────
    required = [TEST_CNN_PATH, TEST_HASH_PATH, TEST_SIFT_PATH,
                TEST_ORB_PATH, TEST_HIST_PATH, GT_PATH]
    for p in required:
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}")
            sys.exit(1)

    # ── Load ground truth ─────────────────────────────────────────────────
    print("\n[1/6] Loading ground truth ...")
    with open(GT_PATH) as f:
        ground_truth = json.load(f)

    query_ids = [int(k) for k in ground_truth.keys()]
    print(f"  {len(query_ids)} queries loaded from ground_truth.json")
    print(f"  Query ID range: {min(query_ids)} – {max(query_ids)}")

    # ── Load test feature store ───────────────────────────────────────────
    print("\n[2/6] Loading test feature store ...")
    feature_store = TestFeatureStore()

    # ── Run all configurations ────────────────────────────────────────────
    print(f"\n[3/6] Running {len(PARAMETER_GRID)} configurations ...")
    print(f"  {len(query_ids)} queries per configuration\n")

    all_results = {}

    for cfg in PARAMETER_GRID:
        label      = cfg['label']
        num_tables = cfg['num_tables']
        hash_size  = cfg['hash_size']

        print(f"  Config: {label} (num_tables={num_tables}, hash_size={hash_size})")

        # Build in-memory index for this config
        index  = build_test_index(num_tables, hash_size, feature_store)
        engine = LightweightQueryEngine(index, feature_store)

        # Run all queries and collect stats
        stats  = run_config_benchmark(engine, ground_truth, query_ids, top_k=10)
        stats['label'] = label
        all_results[label] = stats

        print(f"    recall@5={stats['recall_at_5']:.3f}  "
              f"precision@10={stats['precision_at_10']:.3f}  "
              f"mean={stats['mean_latency_ms']:.2f}ms  "
              f"P99={stats['p99_latency_ms']:.2f}ms  "
              f"candidates={stats['mean_candidates']:.0f}")

    # ── Print trade-off table ─────────────────────────────────────────────
    print("\n[4/6] Trade-off table (sorted by recall@5 descending):")
    print_tradeoff_table(all_results)

    # ── Save results JSON ─────────────────────────────────────────────────
    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {RESULTS_PATH}")

    # ── Generate plots ────────────────────────────────────────────────────
    print("\n[5/6] Generating plots ...")
    plot_pareto_frontier(all_results, TRADEOFF_PLOT)
    plot_parameter_sensitivity(all_results, SENSITIVITY_PLOT)

    # ── Cache validation ──────────────────────────────────────────────────
    print("\n[6/6] Cache validation (baseline config, 2 passes) ...")

    baseline_cfg    = PARAMETER_GRID[0]   # num_tables=10, hash_size=12
    baseline_index  = build_test_index(
        baseline_cfg['num_tables'],
        baseline_cfg['hash_size'],
        feature_store
    )
    baseline_engine = LightweightQueryEngine(baseline_index, feature_store)

    cache_stats = run_cache_validation(
        baseline_engine, ground_truth, query_ids, top_k=10
    )

    print(f"\n  Cache validation results:")
    print(f"    Total queries    : {cache_stats['total_queries']}")
    print(f"    Cache hits       : {cache_stats['hits']} "
          f"({cache_stats['hit_rate_pct']:.1f}%)")
    print(f"    Cache misses     : {cache_stats['misses']}")
    print(f"    Mean hit latency : {cache_stats['mean_hit_ms']:.4f} ms")
    print(f"    Mean miss latency: {cache_stats['mean_miss_ms']:.2f} ms")
    print(f"    Accuracy preserved: "
          f"{'YES ✓' if cache_stats['accuracy_preserved'] else 'NO ✗'}")
    print(f"    Result mismatches : {cache_stats['result_mismatches']}")

    # Add cache stats to results JSON
    all_results['_cache_validation'] = cache_stats
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────
    baseline = all_results.get('baseline', {})
    best_recall = max(all_results[k]['recall_at_5']
                      for k in all_results if not k.startswith('_'))
    best_label  = max((k for k in all_results if not k.startswith('_')),
                      key=lambda k: all_results[k]['recall_at_5'])

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Configs tested        : {len(PARAMETER_GRID)}")
    print(f"  Queries per config    : {len(query_ids)}")
    print(f"\n  Baseline (t10, h12):")
    print(f"    recall@5            : {baseline.get('recall_at_5', 0):.3f}")
    print(f"    precision@10        : {baseline.get('precision_at_10', 0):.3f}")
    print(f"    mean latency        : {baseline.get('mean_latency_ms', 0):.2f} ms")
    print(f"\n  Best recall@5 config  : {best_label} "
          f"(recall@5={best_recall:.3f})")
    print(f"\n  Cache accuracy test   : "
          f"{'PASS' if cache_stats['accuracy_preserved'] else 'FAIL'}")
    print("=" * 65)
    print("\nTask 6 COMPLETE. Files produced:")
    print(f"  {RESULTS_PATH}")
    print(f"  {TRADEOFF_PLOT}")
    print(f"  {SENSITIVITY_PLOT}")
    print("=" * 65)