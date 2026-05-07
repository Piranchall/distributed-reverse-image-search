"""
query_processor.py — Task 3: Distributed Query Processing
Project 20: Distributed Reverse Image Search Engine — Milestone 2
Author: Mahnoor

What this file does:
  1. Loads all 10 LSH hash tables into memory at startup
  2. Given a query embedding, fans out across all 10 tables in parallel (threads)
  3. Collects and deduplicates candidate image IDs
  4. Returns a set of candidate IDs capped at MAX_CANDIDATES

Usage:
    from query_engine.query_processor import QueryProcessor

    qp = QueryProcessor(index_dir='lsh_index', data_path='data/all_cnn_embeddings.h5')
    candidates = qp.get_candidates(query_embedding)   # numpy array (128,)
    print(f"{len(candidates)} candidates found")
"""

import os
import sys
import json
import pickle
import time
import h5py
import numpy as np
import concurrent.futures

# ── Allow import of lsh_structure from sibling folder ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lsh_index.lsh_structure import LSHConfig, hash_embedding


# =============================================================================
# QueryProcessor
# =============================================================================

class QueryProcessor:
    """
    Loads all LSH index tables into memory and answers candidate-lookup queries.

    The 'distributed' aspect is fan-out parallelism: all NUM_TABLES tables are
    queried concurrently using threads so the total lookup time is roughly
    equal to one table lookup, not NUM_TABLES lookups in series.

    Startup cost: ~500–700 MB RAM to hold all 10 tables.
    Per-query cost: one dict key lookup per table (O(1) each), run in parallel.

    Args:
        index_dir  : path to the folder containing lsh_config.json,
                     projection_matrices.pkl, and hash_tables/table_N.pkl
        data_path  : path to all_cnn_embeddings.h5  (for prepare_query_embedding)
    """

    def __init__(self, index_dir: str, data_path: str):
        self.index_dir = index_dir
        self.data_path = data_path

        # ── Load config ───────────────────────────────────────────────────
        config_path = os.path.join(index_dir, 'lsh_config.json')
        with open(config_path, 'r') as f:
            self.config = LSHConfig.from_dict(json.load(f))

        print(f"Config loaded: {self.config.num_tables} tables, "
              f"hash_size={self.config.hash_size}, "
              f"max_candidates={self.config.max_candidates}")

        # ── Load projection matrices ──────────────────────────────────────
        matrices_path = os.path.join(index_dir, 'projection_matrices.pkl')
        with open(matrices_path, 'rb') as f:
            self.projection_matrices = pickle.load(f)

        assert len(self.projection_matrices) == self.config.num_tables, (
            f"Expected {self.config.num_tables} matrices, "
            f"got {len(self.projection_matrices)}"
        )
        print(f"Projection matrices loaded: {len(self.projection_matrices)} × "
              f"{self.projection_matrices[0].shape}")

        # ── Load all hash tables into memory ──────────────────────────────
        # CRITICAL: load once at startup, hold for the process lifetime.
        # Loading from disk per query would add ~100ms I/O latency each time.
        tables_dir = os.path.join(index_dir, 'hash_tables')
        self.hash_tables = []

        print(f"Loading {self.config.num_tables} hash tables from {tables_dir} ...")
        t0 = time.time()
        for i in range(self.config.num_tables):
            table_path = os.path.join(tables_dir, f'table_{i}.pkl')
            with open(table_path, 'rb') as f:
                table = pickle.load(f)
            self.hash_tables.append(table)
            total_ids = sum(len(v) for v in table.values())
            print(f"  table_{i}: {len(table):,} buckets, {total_ids:,} image ID entries")

        elapsed = time.time() - t0
        print(f"All tables loaded in {elapsed:.1f}s\n")
        from incremental_update.concurrent_index import wrap_tables
        self.hash_tables = wrap_tables(self.hash_tables)
        print("Hash tables wrapped in ConcurrentHashTable (M3 Task 2).\n")

        # ── Build id → h5 row index mapping for prepare_query_embedding ──
        print("Building id→index mapping from HDF5 ...")
        with h5py.File(data_path, 'r') as f:
            image_ids = f['image_ids'][:]
        self.id_to_index = {int(gid): idx for idx, gid in enumerate(image_ids)}
        print(f"  Mapping built: {len(self.id_to_index):,} entries\n")

        # ── Timing log for get_query_stats() ─────────────────────────────
        self.query_times_ms = []

    # =========================================================================
    # Core lookup — one table
    # =========================================================================

    def _query_single_table(self, query_embedding: np.ndarray, table_index: int) -> list:
        """
        Hash the query into one table and return the matching bucket's image IDs.

        Args:
            query_embedding: numpy array (128,) float32, already normalized
            table_index:     which hash table to look up (0 … num_tables-1)

        Returns:
            List of integer image IDs in the matching bucket (may be empty).
        """
        bucket_key = hash_embedding(
            query_embedding,
            self.projection_matrices[table_index]
        )
        return self.hash_tables[table_index].get(bucket_key, [])

    # =========================================================================
    # Parallel fan-out — all tables simultaneously
    # =========================================================================

    def get_candidates(self, query_embedding: np.ndarray) -> set:
        """
        Fan out across all NUM_TABLES tables in parallel and collect candidate IDs.

        Uses ThreadPoolExecutor (not ProcessPool) because:
          - Each lookup is a single dict key access — I/O bound on in-memory data
          - Thread overhead is much lower than process-spawn overhead
          - No GIL problem: dict lookups are O(1) and the parallelism gains
            come from overlapping Python thread scheduling

        Applies MAX_CANDIDATES cap deterministically:
          If total candidates > max_candidates, take evenly from each table's
          result list (not random sampling) for reproducibility.

        Args:
            query_embedding: numpy array (128,) float32

        Returns:
            Python set of integer image IDs, size ≤ max_candidates.
        """
        t_start = time.time()

        # Submit one lookup per table, all running concurrently
        per_table_results = [None] * self.config.num_tables

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.num_tables
        ) as executor:
            futures = {
                executor.submit(self._query_single_table, query_embedding, i): i
                for i in range(self.config.num_tables)
            }
            for future in concurrent.futures.as_completed(futures):
                table_idx = futures[future]
                per_table_results[table_idx] = future.result()

        # ── Merge and deduplicate ─────────────────────────────────────────
        candidate_set = set()
        for id_list in per_table_results:
            candidate_set.update(id_list)

        # ── Apply MAX_CANDIDATES cap — deterministic, even across tables ──
        if len(candidate_set) > self.config.max_candidates:
            cap_per_table = self.config.max_candidates // self.config.num_tables
            capped = set()
            for id_list in per_table_results:
                capped.update(id_list[:cap_per_table])
            # Fill remaining slots from any leftover to hit exactly max_candidates
            remaining = self.config.max_candidates - len(capped)
            if remaining > 0:
                extras = candidate_set - capped
                capped.update(list(extras)[:remaining])
            candidate_set = capped

        # ── Record timing for diagnostics ─────────────────────────────────
        elapsed_ms = (time.time() - t_start) * 1000
        self.query_times_ms.append(elapsed_ms)

        return candidate_set

    # =========================================================================
    # Query preparation — handles both in-dataset and new images
    # =========================================================================

    def prepare_query_embedding(self,
                                image_id: int = None,
                                raw_embedding: np.ndarray = None) -> np.ndarray:
        """
        Return a normalized 128-dim float32 embedding ready for get_candidates().

        Two modes:
          - image_id given:      load embedding from HDF5 using id_to_index map
          - raw_embedding given: use directly (new image not in dataset)

        Normalization is CRITICAL: cosine similarity in rank_candidates()
        requires unit vectors. Always normalize here so all downstream callers
        receive consistent input regardless of which mode was used.

        Args:
            image_id:      integer global image ID (image must be in dataset)
            raw_embedding: numpy array (128,) float32 for a new/unseen image

        Returns:
            numpy array (128,) float32, unit length (L2 norm = 1.0)

        Raises:
            ValueError if neither or both arguments are provided.
        """
        if image_id is None and raw_embedding is None:
            raise ValueError("Provide either image_id or raw_embedding, not neither.")
        if image_id is not None and raw_embedding is not None:
            raise ValueError("Provide either image_id or raw_embedding, not both.")

        if image_id is not None:
            if image_id not in self.id_to_index:
                raise KeyError(f"image_id {image_id} not found in the dataset.")
            with h5py.File(self.data_path, 'r') as f:
                row_idx   = self.id_to_index[image_id]
                embedding = f['embeddings'][row_idx].astype(np.float32)
        else:
            embedding = np.array(raw_embedding, dtype=np.float32).flatten()
            if embedding.shape != (self.config.embedding_dim,):
                raise ValueError(
                    f"raw_embedding must be shape ({self.config.embedding_dim},), "
                    f"got {embedding.shape}"
                )

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def get_query_stats(self) -> dict:
        """
        Return latency percentile stats across all queries made so far.

        Called by Hayatullah's load balancer to diagnose performance.
        Safe to call at any time — does not affect query processing.

        Returns:
            dict with keys: total_queries, mean_latency_ms,
                            p50_latency_ms, p95_latency_ms, p99_latency_ms
        """
        if not self.query_times_ms:
            return {
                "total_queries":   0,
                "mean_latency_ms": 0.0,
                "p50_latency_ms":  0.0,
                "p95_latency_ms":  0.0,
                "p99_latency_ms":  0.0,
            }

        times = np.array(self.query_times_ms)
        return {
            "total_queries":   len(times),
            "mean_latency_ms": float(np.mean(times)),
            "p50_latency_ms":  float(np.percentile(times, 50)),
            "p95_latency_ms":  float(np.percentile(times, 95)),
            "p99_latency_ms":  float(np.percentile(times, 99)),
        }
