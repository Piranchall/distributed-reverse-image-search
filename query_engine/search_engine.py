"""
search_engine.py — Task 4: Result Aggregation and Re-ranking
Project 20: Distributed Reverse Image Search Engine — Milestone 2
Author: Mahnoor

What this file does:
  1. Wraps QueryProcessor (Task 3) with the full re-ranking pipeline
  2. Initialises All_Features (FeatureStore) from Milestone 1's feature_fusion.py
  3. Calls rank_candidates() to score candidates using the full weighted metric:
       60% CNN + 20% hash + 10% SIFT + 5% ORB + 5% histogram
  4. Returns final top-K (image_id, score) tuples to the caller

Output contract (for Aliza's Task 6 cache wrapper):
    search(query_embedding, top_k=10)
        → list of (image_id: int, score: float), sorted descending by score

Usage:
    from query_engine.search_engine import SearchEngine

    engine = SearchEngine(
        index_dir  = 'lsh_index',
        data_dir   = 'data',
    )
    results = engine.search(query_embedding, top_k=10)
    for image_id, score in results:
        print(f"  ID={image_id}  score={score:.4f}")
"""

import os
import sys
import json
import time
import h5py
import numpy as np

# ── Allow imports from sibling folders regardless of working directory ────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

# ── Task 3 ────────────────────────────────────────────────────────────────────
from query_engine.query_processor import QueryProcessor

# ── Milestone 1 fusion module ─────────────────────────────────────────────────
# CRITICAL: import from the original location — never copy fusion.py.
# If Piranchal fixes a bug, this import gets the fix automatically.
from feature_Fusion.feature_fusion import All_Features, rank_candidates


# =============================================================================
# SearchEngine
# =============================================================================

class SearchEngine:
    """
    Full search pipeline: LSH candidate lookup → re-ranking → top-K results.

    Startup RAM usage (approximate):
        all_cnn_embeddings.h5      ~512 MB
        all_perceptual_hashes.pkl   ~80 MB
        all_sift_features.pkl      ~150 MB
        all_orb_features.pkl        ~40 MB
        all_hist_features.pkl       ~50 MB
        all_image_paths.pkl         ~20 MB
        10 LSH hash tables         ~600 MB
        ─────────────────────────────────
        Total peak                ~1.4 GB   (requires ≥8 GB machine)

    Args:
        index_dir : folder with lsh_config.json, projection_matrices.pkl,
                    and hash_tables/table_N.pkl
        data_dir  : folder containing all merged feature files from Milestone 1
    """

    def __init__(self, index_dir: str, data_dir: str):
        self.index_dir = index_dir
        self.data_dir  = data_dir

        # ── Resolve feature file paths ────────────────────────────────────
        cnn_path   = os.path.join(data_dir, 'all_cnn_embeddings.h5')
        hash_path  = os.path.join(data_dir, 'all_perceptual_hashes.pkl')
        sift_path  = os.path.join(data_dir, 'all_sift_features.pkl')
        orb_path   = os.path.join(data_dir, 'all_orb_features.pkl')
        hist_path  = os.path.join(data_dir, 'all_hist_features.pkl')
        paths_path = os.path.join(data_dir, 'all_image_paths.pkl')

        for p in [cnn_path, hash_path, sift_path, orb_path, hist_path, paths_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Required feature file not found: {p}\n"
                    f"Download all merged Milestone 1 outputs to {data_dir}/"
                )

        # ── Task 3: QueryProcessor ────────────────────────────────────────
        print("=" * 60)
        print("Initialising QueryProcessor (Task 3) ...")
        print("=" * 60)
        self.query_processor = QueryProcessor(
            index_dir = index_dir,
            data_path = cnn_path,
        )

        # ── Milestone 1: All_Features (FeatureStore) ──────────────────────
        print("=" * 60)
        print("Initialising All_Features from Milestone 1 ...")
        print("=" * 60)
        self.feature_store = All_Features(
            cnn_path   = cnn_path,
            hash_path  = hash_path,
            sift_path  = sift_path,
            orb_path   = orb_path,
            hist_path  = hist_path,
            paths_path = paths_path,
        )

        # ── Timing log ───────────────────────────────────────────────────
        self._search_logs = []   # list of dicts, one per search() call

        print("=" * 60)
        print("SearchEngine ready.")
        print("=" * 60)

    # =========================================================================
    # Main search method
    # =========================================================================

    def search(self,
               query_embedding: np.ndarray,
               query_image_id: int = None,
               top_k: int = None) -> list:
        """
        Full search pipeline for one query.

        Phase 1 — LSH lookup (Task 3):
            Hash query_embedding into all 10 tables in parallel.
            Collect candidate image IDs (typically 500–5000).

        Phase 2 — Re-ranking (Task 4):
            Call rank_candidates() from fusion.py to score each candidate
            using the full weighted metric (60% CNN, 20% hash, 10% SIFT,
            5% ORB, 5% histogram).

        Two modes:
          - query_image_id provided: full fusion scoring (all 5 feature types)
          - query_image_id=None:     CNN cosine similarity only (new image
            not in dataset — full fusion requires hash/SIFT/ORB/hist lookups
            which only work for indexed images)

        Args:
            query_embedding: numpy array (128,) float32, normalized to unit length
            query_image_id:  integer global image ID if image is in the dataset,
                             or None for a new/unseen image
            top_k:           number of results to return
                             (defaults to config.top_k = 10)

        Returns:
            list of (image_id: int, score: float), sorted descending by score.
            Empty list if no candidates found.

        Output contract for Aliza's Task 6:
            Always returns a list of (int, float) tuples, sorted best-first.
        """
        if top_k is None:
            top_k = self.query_processor.config.top_k

        t_total_start = time.time()

        # ── Phase 1: LSH candidate lookup ─────────────────────────────────
        t_lsh_start  = time.time()
        candidate_set = self.query_processor.get_candidates(query_embedding)
        t_lsh_end    = time.time()
        lsh_ms       = (t_lsh_end - t_lsh_start) * 1000

        # ── Edge case: empty candidate set ────────────────────────────────
        if not candidate_set:
            print(f"WARNING: No candidates found for this query. "
                  f"The query embedding may be in an empty bucket across all "
                  f"{self.query_processor.config.num_tables} tables.")
            self._search_logs.append({
                "candidate_count": 0,
                "lsh_ms":          lsh_ms,
                "rerank_ms":       0.0,
                "total_ms":        lsh_ms,
                "top_k":           top_k,
            })
            return []

        # ── Phase 2: Re-ranking ───────────────────────────────────────────
        t_rerank_start = time.time()
        candidate_list = list(candidate_set)

        if query_image_id is not None:
            # Full 5-feature fusion via rank_candidates() from fusion.py
            # rank_candidates() expects a list, not a set
            results = rank_candidates(
                query_image_id,
                candidate_list,
                self.feature_store,
                top_k
            )
        else:
            # New image not in dataset — CNN cosine similarity only
            # (hash/SIFT/ORB/hist lookups require a known image_id)
            results = self._rank_by_cnn_only(query_embedding, candidate_list, top_k)

        t_rerank_end = time.time()
        rerank_ms    = (t_rerank_end - t_rerank_start) * 1000
        total_ms     = (t_rerank_end - t_total_start) * 1000

        # ── Edge case: fewer candidates than top_k ────────────────────────
        # rank_candidates() already handles this — it returns all candidates
        # if len(candidates) < top_k. No padding needed.

        # ── Log timing ────────────────────────────────────────────────────
        self._search_logs.append({
            "candidate_count": len(candidate_set),
            "lsh_ms":          round(lsh_ms, 2),
            "rerank_ms":       round(rerank_ms, 2),
            "total_ms":        round(total_ms, 2),
            "top_k":           top_k,
        })

        return results

    # =========================================================================
    # CNN-only fallback for new/unseen images
    # =========================================================================

    def _rank_by_cnn_only(self,
                          query_embedding: np.ndarray,
                          candidate_ids: list,
                          top_k: int) -> list:
        """
        CNN cosine similarity ranking for images not in the dataset.

        Used when query_image_id=None (new image). Full fusion is unavailable
        because hash/SIFT/ORB/hist lookups require a known dataset image_id.

        Args:
            query_embedding: normalized numpy (128,) float32
            candidate_ids:   list of integer image IDs to score
            top_k:           how many to return

        Returns:
            list of (image_id, float_score) sorted descending
        """
        if not candidate_ids:
            return []

        # Stack all candidate CNN embeddings: (N, 128)
        candidate_matrix = np.stack(
            [self.feature_store.get_cnn(cid).astype(np.float32)
             for cid in candidate_ids],
            axis=0
        )

        # Normalize each candidate row
        norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1e-8, norms)
        candidate_matrix = candidate_matrix / norms

        # Cosine similarities in one matrix multiply
        scores = candidate_matrix @ query_embedding   # shape (N,)

        # Sort descending, return top_k
        ranked = sorted(
            zip(candidate_ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_search_stats(self) -> dict:
        """
        Return percentile timing stats across all search() calls made so far.

        Structure matches QueryProcessor.get_query_stats() for consistency.
        Called by Hayatullah's benchmark and Aliza's cache for estimated time saved.

        Returns:
            dict with keys: total_queries, mean_latency_ms,
                            p50_latency_ms, p95_latency_ms, p99_latency_ms,
                            mean_lsh_ms, mean_rerank_ms, mean_candidates
        """
        if not self._search_logs:
            return {k: 0.0 for k in [
                "total_queries", "mean_latency_ms",
                "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                "mean_lsh_ms", "mean_rerank_ms", "mean_candidates",
            ]}

        total_ms   = np.array([l["total_ms"]        for l in self._search_logs])
        lsh_ms     = np.array([l["lsh_ms"]           for l in self._search_logs])
        rerank_ms  = np.array([l["rerank_ms"]        for l in self._search_logs])
        candidates = np.array([l["candidate_count"]  for l in self._search_logs])

        return {
            "total_queries":   len(self._search_logs),
            "mean_latency_ms": float(np.mean(total_ms)),
            "p50_latency_ms":  float(np.percentile(total_ms, 50)),
            "p95_latency_ms":  float(np.percentile(total_ms, 95)),
            "p99_latency_ms":  float(np.percentile(total_ms, 99)),
            "mean_lsh_ms":     float(np.mean(lsh_ms)),
            "mean_rerank_ms":  float(np.mean(rerank_ms)),
            "mean_candidates": float(np.mean(candidates)),
        }


# =============================================================================
# Integration test — run directly with: python query_engine/search_engine.py
# =============================================================================

if __name__ == '__main__':
    import json
    import random

    # ── Paths — adjust DATA_DIR to wherever your merged files are ────────
    INDEX_DIR = 'lsh_index'
    DATA_DIR  = 'data'
    GT_PATH   = 'validation/ground_truth.json'

    print("=" * 60)
    print("SearchEngine — Integration Test")
    print("=" * 60)

    # ── Instantiate engine ────────────────────────────────────────────────
    engine = SearchEngine(index_dir=INDEX_DIR, data_dir=DATA_DIR)

    # ── Load ground truth ─────────────────────────────────────────────────
    if not os.path.exists(GT_PATH):
        print(f"\nWARNING: {GT_PATH} not found — running smoke test with random IDs instead.")
        test_ids = random.sample(list(engine.feature_store.id_to_index.keys()), 10)
        ground_truth = {str(qid): [] for qid in test_ids}
    else:
        with open(GT_PATH, 'r') as f:
            ground_truth = json.load(f)
        test_ids = [int(k) for k in list(ground_truth.keys())[:10]]
        print(f"Ground truth loaded: {len(ground_truth)} base images")

    # ── Run 10 test queries ───────────────────────────────────────────────
    print(f"\nRunning {len(test_ids)} test queries ...\n")

    hits_at_10    = 0
    total_variants = 0

    for query_id in test_ids:
        # Prepare normalized embedding
        query_emb = engine.query_processor.prepare_query_embedding(image_id=query_id)

        # Full search with fusion re-ranking
        results = engine.search(
            query_embedding = query_emb,
            query_image_id  = query_id,
            top_k           = 10,
        )

        result_ids   = [r[0] for r in results]
        known_variants = [int(v) for v in ground_truth.get(str(query_id), [])]
        found        = [v for v in known_variants if v in result_ids]

        hits_at_10    += len(found)
        total_variants += len(known_variants)

        log = engine._search_logs[-1]
        print(f"  Query ID {query_id:>7d} | "
              f"candidates={log['candidate_count']:>5d} | "
              f"LSH={log['lsh_ms']:>6.1f}ms | "
              f"rerank={log['rerank_ms']:>6.1f}ms | "
              f"total={log['total_ms']:>6.1f}ms | "
              f"variants found: {len(found)}/{len(known_variants)}")

    # ── Summary ───────────────────────────────────────────────────────────
    stats = engine.get_search_stats()
    precision = hits_at_10 / (len(test_ids) * 10) if test_ids else 0
    recall    = hits_at_10 / total_variants if total_variants else 0

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  Queries run        : {stats['total_queries']}")
    print(f"  Mean total latency : {stats['mean_latency_ms']:.1f} ms")
    print(f"  P50 latency        : {stats['p50_latency_ms']:.1f} ms")
    print(f"  P95 latency        : {stats['p95_latency_ms']:.1f} ms")
    print(f"  Mean LSH time      : {stats['mean_lsh_ms']:.1f} ms")
    print(f"  Mean rerank time   : {stats['mean_rerank_ms']:.1f} ms")
    print(f"  Mean candidates    : {stats['mean_candidates']:.0f}")
    print(f"  Precision@10       : {precision:.3f}  (target > 0.5)")
    print(f"  Recall@10          : {recall:.3f}  (target > 0.6)")

    if stats['mean_latency_ms'] > 500:
        print("\n  WARNING: Mean latency > 500ms — may need index tuning.")
    if recall < 0.3 and total_variants > 0:
        print("\n  WARNING: Low recall — consider reducing HASH_SIZE or "
              "increasing NUM_TABLES in lsh_config.json.")
    print("=" * 60)
