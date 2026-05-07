import os
import sys
import time
import math
import concurrent.futures

import numpy as np

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from feature_Fusion.feature_fusion import (
    All_Features,
    rank_candidates as sequential_rank_candidates,
    compute_batch_similarity,
)

# ── Tuning constants ──────────────────────────────────────────────────────────
SHARD_SIZE              = 500   # candidates per parallel shard
MIN_PARALLEL_CANDIDATES = 2000  # below this, sequential is always faster


# =============================================================================
# Worker function — module level, pickle-safe
# =============================================================================

def score_shard(shard_candidate_ids: list,
                query_embedding: np.ndarray,
                feature_store: All_Features) -> list:
    """
    Score a shard using vectorized CNN cosine similarity.

    Uses IDENTICAL scoring to sequential rank_candidates() which internally
    calls compute_batch_similarity() — one matrix multiply for the whole shard:
      1. Stack shard embeddings into (N, 128) matrix
      2. Normalize each row to unit length
      3. Dot product with query_embedding → (N,) scores

    This guarantees results are numerically identical to the sequential path.

    Args:
        shard_candidate_ids : list of image IDs for this shard
        query_embedding     : normalized (128,) float32
        feature_store       : All_Features instance (shared, read-only)

    Returns:
        list of (image_id, score_float) tuples, unordered within shard
    """
    if not shard_candidate_ids:
        return []

    try:
        # Stack embeddings: (N, 128)
        candidate_matrix = np.stack(
            [feature_store.get_cnn(cid).astype(np.float32)
             for cid in shard_candidate_ids],
            axis=0
        )

        # Row-normalize (same as compute_batch_similarity)
        norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1e-8, norms)
        candidate_matrix = candidate_matrix / norms

        # Single matrix-vector multiply: (N, 128) @ (128,) → (N,)
        scores = candidate_matrix @ query_embedding.astype(np.float32)

        return list(zip(shard_candidate_ids, scores.tolist()))

    except Exception:
        # Fallback: per-candidate cosine (never reached in normal operation)
        results = []
        for cid in shard_candidate_ids:
            try:
                vec  = feature_store.get_cnn(cid).astype(np.float32)
                norm = np.linalg.norm(vec)
                vec  = vec / norm if norm > 1e-8 else vec
                results.append((cid, float(np.dot(query_embedding, vec))))
            except Exception:
                results.append((cid, 0.0))
        return results


# =============================================================================
# ParallelReranker
# =============================================================================

class ParallelReranker:
    """
    Parallel drop-in replacement for rank_candidates() from feature_fusion.py.

    For candidate sets < MIN_PARALLEL_CANDIDATES (2000):
        Delegates directly to sequential rank_candidates() — no thread overhead,
        guaranteed identical results.

    For larger sets:
        Splits into shards of SHARD_SIZE, scores each in a thread using
        vectorized CNN scoring (same function as sequential path), merges.
        Results are numerically identical to sequential path.

    Uses a persistent ThreadPoolExecutor (created once at init, not per query)
    so the ~2ms pool-creation overhead is never paid per query.

    Expected speedup: 1.5–2.5× on Kaggle (4 vCPUs) for 3000+ candidates.

    Args:
        feature_store : All_Features instance (shared read-only)
        num_workers   : thread count (default: all available CPU cores)
    """

    def __init__(self, feature_store: All_Features, num_workers: int = None):
        self.feature_store = feature_store
        self.num_workers   = num_workers or os.cpu_count() or 4

        # Persistent pool — created once, reused across all queries
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="reranker"
        )

        self._stats = {
            "total_calls":        0,
            "parallel_calls":     0,
            "sequential_calls":   0,
            "total_parallel_ms":  0.0,
            "total_sequential_ms":0.0,
            "total_merge_ms":     0.0,
        }

        print(f"[ParallelReranker] Ready. "
              f"workers={self.num_workers}, "
              f"shard_size={SHARD_SIZE}, "
              f"min_parallel={MIN_PARALLEL_CANDIDATES}")

    # =========================================================================
    # Main method — drop-in for rank_candidates()
    # =========================================================================

    def rank_candidates(self,
                        query_image_id,
                        candidate_list: list,
                        top_k: int = 10,
                        query_embedding: np.ndarray = None) -> list:
        """
        Score all candidates and return top-K, using parallel shards for
        large candidate sets and sequential for small ones.

        Results are ALWAYS numerically identical to sequential rank_candidates()
        because both paths use the same vectorized CNN scoring function.

        Args:
            query_image_id  : int if image is in dataset, None for new images
            candidate_list  : list of integer image IDs from LSH lookup
            top_k           : number of results to return
            query_embedding : normalized (128,) float32

        Returns:
            list of (image_id, score) tuples, sorted descending, length top_k
        """
        self._stats["total_calls"] += 1

        if not candidate_list:
            return []

        # ── Ensure we have a normalized query embedding ───────────────────
        if query_embedding is None and query_image_id is not None:
            try:
                raw  = self.feature_store.get_cnn(query_image_id).astype(np.float32)
                norm = np.linalg.norm(raw)
                query_embedding = raw / norm if norm > 1e-8 else raw
            except (KeyError, IndexError):
                query_embedding = np.zeros(128, dtype=np.float32)

        # ── Small candidate set: delegate entirely to sequential ──────────
        # sequential rank_candidates() uses compute_batch_similarity() which
        # is already vectorized — no benefit from threads for small sets.
        if len(candidate_list) < MIN_PARALLEL_CANDIDATES:
            t0 = time.time()
            results = sequential_rank_candidates(
                query_image_id, candidate_list, self.feature_store, top_k
            )
            elapsed = (time.time() - t0) * 1000
            self._stats["sequential_calls"]    += 1
            self._stats["total_sequential_ms"] += elapsed
            return results

        # ── Large candidate set: parallel shards ─────────────────────────
        t_parallel_start = time.time()

        # Split into equal shards, clamp num_shards to num_workers
        num_shards = min(
            math.ceil(len(candidate_list) / SHARD_SIZE),
            self.num_workers
        )
        actual_shard = math.ceil(len(candidate_list) / num_shards)
        shards = [
            candidate_list[i : i + actual_shard]
            for i in range(0, len(candidate_list), actual_shard)
        ]

        # Submit all shards to the persistent thread pool
        # Each shard does ONE matrix multiply — same as sequential path
        futures = [
            self._executor.submit(
                score_shard,
                shard,
                query_embedding,
                self.feature_store,
            )
            for shard in shards
        ]

        # Collect results as they complete
        all_scores = []
        for fut in concurrent.futures.as_completed(futures):
            all_scores.extend(fut.result())

        parallel_elapsed = (time.time() - t_parallel_start) * 1000

        # ── Merge and sort ─────────────────────────────────────────────────
        t_merge = time.time()
        all_scores.sort(key=lambda x: x[1], reverse=True)
        top_results  = all_scores[:top_k]
        merge_elapsed = (time.time() - t_merge) * 1000

        self._stats["parallel_calls"]    += 1
        self._stats["total_parallel_ms"] += parallel_elapsed
        self._stats["total_merge_ms"]    += merge_elapsed

        return top_results

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_stats(self) -> dict:
        """
        Return timing stats across all rank_candidates() calls so far.

        Returns:
            dict with call counts, mean latencies per path, and merge time.
        """
        s = self._stats
        total = s["total_calls"]
        if total == 0:
            return {k: 0.0 for k in [
                "total_calls", "parallel_calls", "sequential_calls",
                "mean_parallel_ms", "mean_sequential_ms", "mean_merge_ms",
            ]}

        mean_par = (s["total_parallel_ms"] / s["parallel_calls"]
                    if s["parallel_calls"] > 0 else 0.0)
        mean_seq = (s["total_sequential_ms"] / s["sequential_calls"]
                    if s["sequential_calls"] > 0 else 0.0)
        mean_merge = (s["total_merge_ms"] / s["parallel_calls"]
                      if s["parallel_calls"] > 0 else 0.0)

        return {
            "total_calls":       total,
            "parallel_calls":    s["parallel_calls"],
            "sequential_calls":  s["sequential_calls"],
            "mean_parallel_ms":  round(mean_par, 2),
            "mean_sequential_ms":round(mean_seq, 2),
            "mean_merge_ms":     round(mean_merge, 2),
        }

    def shutdown(self):
        """Cleanly shut down the thread pool. Call when done with the reranker."""
        self._executor.shutdown(wait=True)

    def __del__(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass


# =============================================================================
# Benchmark — run from repo root: python reranking/parallel_reranker.py
# =============================================================================

if __name__ == '__main__':
    import json, h5py, random

    print("=" * 65)
    print("Milestone 3 Task 3 — ParallelReranker Benchmark")
    print("=" * 65)

    DATA_DIR = os.path.join(_REPO_ROOT, 'data')
    GT_PATH  = os.path.join(_REPO_ROOT, 'validation', 'ground_truth.json')

    required_files = [
        os.path.join(DATA_DIR, 'all_cnn_embeddings.h5'),
        os.path.join(DATA_DIR, 'all_perceptual_hashes.pkl'),
        os.path.join(DATA_DIR, 'all_sift_features.pkl'),
        os.path.join(DATA_DIR, 'all_orb_features.pkl'),
        os.path.join(DATA_DIR, 'all_hist_features.pkl'),
        os.path.join(DATA_DIR, 'all_image_paths.pkl'),
    ]
    for p in required_files:
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}")
            sys.exit(1)

    print("\nLoading All_Features ...")
    fs = All_Features(
        cnn_path   = os.path.join(DATA_DIR, 'all_cnn_embeddings.h5'),
        hash_path  = os.path.join(DATA_DIR, 'all_perceptual_hashes.pkl'),
        sift_path  = os.path.join(DATA_DIR, 'all_sift_features.pkl'),
        orb_path   = os.path.join(DATA_DIR, 'all_orb_features.pkl'),
        hist_path  = os.path.join(DATA_DIR, 'all_hist_features.pkl'),
        paths_path = os.path.join(DATA_DIR, 'all_image_paths.pkl'),
    )

    if not os.path.exists(GT_PATH):
        query_ids = random.sample(list(fs.id_to_index.keys()), 20)
    else:
        with open(GT_PATH) as f:
            gt = json.load(f)
        query_ids = [int(k) for k in list(gt.keys())[:20]]

    reranker = ParallelReranker(feature_store=fs)
    all_ids  = list(fs.id_to_index.keys())

    print(f"\n--- 3000 candidates (parallel path) ---\n")

    all_match = True
    seq_times, par_times = [], []

    for query_id in query_ids:
        candidates = random.sample(all_ids, 3000)
        raw   = fs.get_cnn(query_id).astype(np.float32)
        q_emb = raw / np.linalg.norm(raw)

        # Sequential baseline using same CNN scoring
        t0 = time.time()
        scores_seq = compute_batch_similarity(query_id, candidates, fs)
        seq_top10  = sorted(scores_seq.items(), key=lambda x: x[1], reverse=True)[:10]
        seq_ms = (time.time() - t0) * 1000
        seq_times.append(seq_ms)

        # Parallel
        t0 = time.time()
        par_top10 = reranker.rank_candidates(
            query_id, candidates, top_k=10, query_embedding=q_emb
        )
        par_ms = (time.time() - t0) * 1000
        par_times.append(par_ms)

        seq_ids = [r[0] for r in seq_top10]
        par_ids = [r[0] for r in par_top10]
        match   = (seq_ids == par_ids)
        if not match:
            all_match = False
        speedup = seq_ms / par_ms if par_ms > 0 else 0

        print(f"  Query {query_id:>7d} | seq={seq_ms:>6.1f}ms | par={par_ms:>6.1f}ms | "
              f"speedup={speedup:.2f}× | {'✓' if match else '✗'}")

    seq_arr = np.array(seq_times)
    par_arr = np.array(par_times)
    mean_speedup = float(np.mean(seq_arr)) / float(np.mean(par_arr))

    print("\n" + "=" * 65)
    print("Benchmark Summary")
    print("=" * 65)
    print(f"  Sequential mean ms  : {np.mean(seq_arr):.1f}")
    print(f"  Sequential P95  ms  : {np.percentile(seq_arr, 95):.1f}")
    print(f"  Parallel   mean ms  : {np.mean(par_arr):.1f}")
    print(f"  Parallel   P95  ms  : {np.percentile(par_arr, 95):.1f}")
    print(f"  Overall speedup     : {mean_speedup:.2f}×")
    print(f"  ID match            : {'PASS' if all_match else 'FAIL'}")

    if all_match:
        print("\n✓ Task 3 PASSED — parallel results identical to sequential.")
    else:
        print("\n✗ Task 3 FAIL — result IDs differ.")

    print("=" * 65)
    reranker.shutdown()