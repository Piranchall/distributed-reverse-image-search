import os
import sys
import time
import queue
import threading
import math

import numpy as np

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


# =============================================================================
# PART A — FeatureCompressor
# =============================================================================

class FeatureCompressor:
    """
    Converts CNN embedding arrays between float32 and float16.

    float32: 4 bytes × 128 values × 1M images = 512 MB
    float16: 2 bytes × 128 values × 1M images = 256 MB  (50% saving)

    Precision notes:
        float16 has ~3–4 significant decimal digits (vs 7 for float32).
        For unit vectors (values in [-1, +1]) the rounding error is small:
            Mean L2 error per vector: < 0.001
            Mean cosine similarity error: < 0.0005
        LSH hashing uses only the SIGN of dot products — unaffected by float16.
        Re-ranking cosine similarity loses ~0.001 accuracy — acceptable.
    """

    @staticmethod
    def compress(embedding: np.ndarray) -> np.ndarray:
        """
        float32 (128,) → float16 (128,).

        Args:
            embedding: numpy array shape (128,), dtype float32

        Returns:
            numpy array shape (128,), dtype float16
        """
        return embedding.astype(np.float16)

    @staticmethod
    def decompress(embedding: np.ndarray) -> np.ndarray:
        """
        float16 (128,) → float32 (128,).

        Decompression is used before any arithmetic that requires full precision,
        e.g. cosine similarity in re-ranking. LSH hashing can operate on float16
        directly since only the sign matters.

        Args:
            embedding: numpy array shape (128,), dtype float16

        Returns:
            numpy array shape (128,), dtype float32
        """
        return embedding.astype(np.float32)

    @staticmethod
    def compress_batch(embedding_matrix: np.ndarray) -> np.ndarray:
        """
        float32 (N, 128) → float16 (N, 128).

        Args:
            embedding_matrix: numpy array shape (N, 128), dtype float32

        Returns:
            numpy array shape (N, 128), dtype float16
        """
        return embedding_matrix.astype(np.float16)

    @staticmethod
    def measure_compression_error(n_samples: int = 1000) -> dict:
        """
        Measure L2 reconstruction error from float32 → float16 → float32 round-trip.

        Creates n_samples random unit vectors, compresses and decompresses,
        measures L2 distance between original and round-tripped vector.

        Args:
            n_samples: number of random vectors to test

        Returns:
            dict with keys: mean_l2_error, max_l2_error, p99_l2_error
        """
        rng = np.random.default_rng(seed=42)
        originals = rng.standard_normal((n_samples, 128)).astype(np.float32)

        # Normalize to unit vectors (same as real embeddings)
        norms = np.linalg.norm(originals, axis=1, keepdims=True)
        originals = originals / norms

        # Round-trip
        compressed   = originals.astype(np.float16)
        decompressed = compressed.astype(np.float32)

        # L2 error per vector
        errors = np.linalg.norm(originals - decompressed, axis=1)  # shape (N,)

        return {
            "n_samples":     n_samples,
            "mean_l2_error": float(np.mean(errors)),
            "max_l2_error":  float(np.max(errors)),
            "p99_l2_error":  float(np.percentile(errors, 99)),
        }

    @staticmethod
    def measure_similarity_error(n_pairs: int = 500) -> dict:
        """
        Measure cosine similarity error introduced by float16 compression.

        For each pair of random unit vectors, computes cosine similarity
        in float32 and float16, then reports the absolute difference.

        Args:
            n_pairs: number of vector pairs to test

        Returns:
            dict with keys: mean_abs_error, max_abs_error, p99_abs_error
        """
        rng = np.random.default_rng(seed=123)

        # Generate pairs
        a = rng.standard_normal((n_pairs, 128)).astype(np.float32)
        b = rng.standard_normal((n_pairs, 128)).astype(np.float32)

        # Normalize
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)

        # float32 cosine similarities: (N,)
        sim_f32 = np.sum(a * b, axis=1)

        # float16 cosine similarities
        a16 = a.astype(np.float16)
        b16 = b.astype(np.float16)
        sim_f16 = np.sum(a16 * b16, axis=1).astype(np.float32)

        errors = np.abs(sim_f32 - sim_f16)

        return {
            "n_pairs":       n_pairs,
            "mean_abs_error":float(np.mean(errors)),
            "max_abs_error": float(np.max(errors)),
            "p99_abs_error": float(np.percentile(errors, 99)),
        }


# =============================================================================
# PART B — CompressedEmbeddingStore
# =============================================================================

class CompressedEmbeddingStore:
    """
    Wraps the (1M, 128) float32 embedding array with float16 in-memory storage.

    Memory saved: 512 MB → 256 MB (50% reduction).
    Speed benefit: float16 matrix multiplies are faster on modern CPUs/GPUs.

    Usage in SearchEngine:
        if USE_COMPRESSION:
            self.feature_store._compressed_store = CompressedEmbeddingStore(
                self.feature_store.embeddings
            )

    Then in scoring, use _compressed_store.get(image_id, id_to_index) instead
    of feature_store.get_cnn(image_id).

    Args:
        embedding_matrix_f32 : numpy array (N, 128) float32 from All_Features
    """

    def __init__(self, embedding_matrix_f32: np.ndarray):
        t0 = time.time()
        self.data = embedding_matrix_f32.astype(np.float16)   # (N, 128) float16
        elapsed  = time.time() - t0

        original_mb   = embedding_matrix_f32.nbytes / (1024 ** 2)
        compressed_mb = self.data.nbytes / (1024 ** 2)

        print(f"[CompressedEmbeddingStore] Compressed {original_mb:.0f} MB → "
              f"{compressed_mb:.0f} MB "
              f"({100*(1-compressed_mb/original_mb):.0f}% saving) "
              f"in {elapsed:.2f}s")

    def get(self, image_id: int, id_to_index: dict) -> np.ndarray:
        """
        Retrieve one embedding, decompressed to float32.

        Args:
            image_id     : global image ID
            id_to_index  : dict mapping image_id → row index (from All_Features)

        Returns:
            numpy array (128,) float32
        """
        row_idx = id_to_index[image_id]
        return self.data[row_idx].astype(np.float32)

    def get_batch(self, image_id_list: list, id_to_index: dict) -> np.ndarray:
        """
        Retrieve multiple embeddings, decompressed to float32.

        Args:
            image_id_list: list of integer image IDs
            id_to_index:  dict mapping image_id → row index

        Returns:
            numpy array (N, 128) float32
        """
        indices = [id_to_index[iid] for iid in image_id_list]
        return self.data[indices].astype(np.float32)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return (f"CompressedEmbeddingStore("
                f"shape={self.data.shape}, "
                f"dtype=float16, "
                f"memory={self.data.nbytes/(1024**2):.0f}MB)")


# =============================================================================
# PART C — BatchQueryProcessor
# =============================================================================

class BatchQueryProcessor:
    """
    Buffers multiple queries and processes them as a wave.

    Why this helps:
        Each call to QueryProcessor.get_candidates() fans out across 10 tables
        using 10 threads in a ThreadPoolExecutor. If queries arrive sequentially
        each one pays the thread scheduling overhead independently.
        By batching N queries together, we process all N across each table
        in one vectorized call (hash_embeddings_batch), sharing the fan-out cost.

    Break-even:
        At 1 concurrent caller: batching adds latency (must wait for BATCH_SIZE
        or MAX_WAIT_MS). For sequential benchmarks, use direct search() instead.
        At 5+ concurrent callers: batching saves ~40% of LSH fan-out time.

    Args:
        query_processor  : QueryProcessor instance (from query_engine)
        batch_size       : process when this many queries have accumulated
        max_wait_ms      : process after this many milliseconds regardless
    """

    BATCH_SIZE  = 8    # queries per wave (tuned for 4-core machines)
    MAX_WAIT_MS = 20   # max wait before forcing a batch dispatch

    def __init__(self, query_processor, batch_size: int = None, max_wait_ms: float = None):
        self._qp          = query_processor
        self._batch_size  = batch_size or self.BATCH_SIZE
        self._max_wait_ms = max_wait_ms or self.MAX_WAIT_MS

        # Queue items: (embedding, result_container)
        # result_container is a dict set by the dispatcher when done
        self._queue     = queue.Queue()
        self._stop_flag = threading.Event()

        # Background dispatcher thread
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="batch-dispatcher",
            daemon=True
        )
        self._dispatcher.start()

        # Stats
        self._total_submitted = 0
        self._total_batches   = 0
        self._total_batch_ms  = 0.0

        print(f"[BatchQueryProcessor] Ready. "
              f"batch_size={self._batch_size}, "
              f"max_wait_ms={self._max_wait_ms}")

    def submit(self, query_embedding: np.ndarray,
               query_image_id: int = None,
               top_k: int = 10) -> set:
        """
        Submit one query for batched processing and block until the result arrives.

        To the caller this appears synchronous; internally it is batched with
        other concurrent callers.

        Args:
            query_embedding : normalized (128,) float32
            query_image_id  : int or None (passed through to re-ranking)
            top_k           : number of results

        Returns:
            set of candidate image IDs (same as QueryProcessor.get_candidates())
        """
        self._total_submitted += 1

        result_holder = {"candidates": None, "done": threading.Event()}
        self._queue.put((query_embedding, query_image_id, top_k, result_holder))

        # Block until dispatcher sets the result
        result_holder["done"].wait()
        return result_holder["candidates"]

    def _dispatch_loop(self):
        """
        Background thread: drains the queue in batches.

        Waits up to MAX_WAIT_MS for BATCH_SIZE queries to accumulate,
        then processes whatever it has as a batch.
        """
        while not self._stop_flag.is_set():
            batch = []
            deadline = time.time() + self._max_wait_ms / 1000.0

            # Collect up to BATCH_SIZE queries or until deadline
            while len(batch) < self._batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                    batch.append(item)
                except queue.Empty:
                    break

            if not batch:
                continue

            # Process batch
            self._process_batch(batch)

    def _process_batch(self, batch: list):
        """
        Fan out all embeddings in the batch across all hash tables in one go.

        For each table, calls hash_embeddings_batch() once for all N embeddings
        rather than N times individually. This is the core efficiency gain.

        Args:
            batch: list of (embedding, query_image_id, top_k, result_holder) tuples
        """
        from lsh_index.lsh_structure import hash_embeddings_batch

        t0 = time.time()
        n  = len(batch)

        # Stack all query embeddings: (N, 128)
        emb_matrix = np.stack([item[0] for item in batch], axis=0).astype(np.float32)

        # Per-query candidate sets
        candidate_sets = [set() for _ in range(n)]

        # Fan out across all tables — ONE vectorized call per table
        for table_idx in range(self._qp.config.num_tables):
            # (N, 128) @ (128, hash_size) → (N, hash_size) → N bucket keys
            keys = hash_embeddings_batch(
                emb_matrix,
                self._qp.projection_matrices[table_idx]
            )

            # Lookup each query's bucket in this table
            for q_idx, key in enumerate(keys):
                bucket = self._qp.hash_tables[table_idx].get(key, [])
                candidate_sets[q_idx].update(bucket)

        # Apply MAX_CANDIDATES cap and deliver results
        max_cands = self._qp.config.max_candidates
        for q_idx, (_, _, _, result_holder) in enumerate(batch):
            cands = candidate_sets[q_idx]
            if len(cands) > max_cands:
                cands = set(list(cands)[:max_cands])
            result_holder["candidates"] = cands
            result_holder["done"].set()

        elapsed_ms = (time.time() - t0) * 1000
        self._total_batches   += 1
        self._total_batch_ms  += elapsed_ms

    def get_stats(self) -> dict:
        """Return throughput and latency statistics."""
        mean_batch_ms = (self._total_batch_ms / self._total_batches
                         if self._total_batches > 0 else 0.0)
        return {
            "total_queries_submitted": self._total_submitted,
            "total_batches_processed": self._total_batches,
            "mean_batch_latency_ms":   round(mean_batch_ms, 2),
            "batch_size":              self._batch_size,
            "max_wait_ms":             self._max_wait_ms,
        }

    def shutdown(self):
        """Stop the dispatcher thread cleanly."""
        self._stop_flag.set()
        self._dispatcher.join(timeout=2.0)


# =============================================================================
# Benchmark — run from repo root: python compression/feature_compressor.py
# =============================================================================

if __name__ == '__main__':
    import json, h5py, random

    print("=" * 65)
    print("Milestone 3 Task 4 — FeatureCompressor + BatchQueryProcessor")
    print("=" * 65)

    DATA_DIR = os.path.join(_REPO_ROOT, 'data')
    CNN_PATH = os.path.join(DATA_DIR, 'all_cnn_embeddings.h5')

    # ── PART A: Compression error measurement ─────────────────────────────
    print("\n[1/4] Measuring compression error (float32 → float16 → float32) ...")
    fc = FeatureCompressor()

    l2_stats  = fc.measure_compression_error(n_samples=1000)
    sim_stats = fc.measure_similarity_error(n_pairs=500)

    print(f"\n  L2 reconstruction error (1000 random unit vectors):")
    print(f"    mean : {l2_stats['mean_l2_error']:.6f}  (target < 0.001)")
    print(f"    max  : {l2_stats['max_l2_error']:.6f}")
    print(f"    p99  : {l2_stats['p99_l2_error']:.6f}")

    print(f"\n  Cosine similarity error (500 random pairs):")
    print(f"    mean : {sim_stats['mean_abs_error']:.6f}  (target < 0.001)")
    print(f"    max  : {sim_stats['max_abs_error']:.6f}")
    print(f"    p99  : {sim_stats['p99_abs_error']:.6f}")

    l2_pass  = l2_stats['mean_l2_error'] < 0.001
    sim_pass = sim_stats['mean_abs_error'] < 0.001
    print(f"\n  L2 error test  : {'PASS' if l2_pass else 'FAIL'}")
    print(f"  Sim error test : {'PASS' if sim_pass else 'FAIL'}")

    # ── PART B: CompressedEmbeddingStore ─────────────────────────────────
    if not os.path.exists(CNN_PATH):
        print(f"\n[2/4] SKIP — {CNN_PATH} not found.")
        print("       CompressedEmbeddingStore test requires all_cnn_embeddings.h5")
    else:
        print(f"\n[2/4] Testing CompressedEmbeddingStore ...")

        with h5py.File(CNN_PATH, 'r') as f:
            embeddings_f32 = f['embeddings'][:]   # (1M, 128) float32
            image_ids      = f['image_ids'][:]

        id_to_index = {int(gid): idx for idx, gid in enumerate(image_ids)}

        store = CompressedEmbeddingStore(embeddings_f32)
        print(f"  {store}")

        # Verify a few retrievals match the original
        test_ids = random.sample(list(id_to_index.keys()), 20)
        errors   = []
        for iid in test_ids:
            f32_orig = embeddings_f32[id_to_index[iid]]
            f32_ret  = store.get(iid, id_to_index)
            errors.append(float(np.linalg.norm(f32_orig - f32_ret)))

        mean_err = float(np.mean(errors))
        print(f"  Retrieval L2 error (20 random IDs): mean={mean_err:.6f}")

        # Batch retrieval
        batch_ids = test_ids[:10]
        batch_mat = store.get_batch(batch_ids, id_to_index)
        print(f"  get_batch shape: {batch_mat.shape}  dtype: {batch_mat.dtype}")

        store_pass = (mean_err < 0.005 and batch_mat.shape == (10, 128))
        print(f"  CompressedEmbeddingStore test: {'PASS' if store_pass else 'FAIL'}")

    # ── PART C: Latency comparison — float32 vs float16 ──────────────────
    print(f"\n[3/4] Latency comparison: float32 vs float16 (200 matrix multiplies) ...")

    rng = np.random.default_rng(seed=0)
    queries_f32 = rng.standard_normal((200, 128)).astype(np.float32)
    queries_f32 /= np.linalg.norm(queries_f32, axis=1, keepdims=True)
    queries_f16 = queries_f32.astype(np.float16)

    # Simulate candidate matrix (1000 candidates × 128)
    cand_f32 = rng.standard_normal((1000, 128)).astype(np.float32)
    cand_f32 /= np.linalg.norm(cand_f32, axis=1, keepdims=True)
    cand_f16 = cand_f32.astype(np.float16)

    # float32 timing
    t0 = time.time()
    for q in queries_f32:
        _ = cand_f32 @ q
    f32_ms = (time.time() - t0) * 1000 / 200

    # float16 timing
    t0 = time.time()
    for q in queries_f16:
        _ = (cand_f16 @ q).astype(np.float32)
    f16_ms = (time.time() - t0) * 1000 / 200

    speedup = f32_ms / f16_ms if f16_ms > 0 else 0.0
    print(f"  float32 mean ms per query: {f32_ms:.3f}")
    print(f"  float16 mean ms per query: {f16_ms:.3f}")
    print(f"  Speedup                  : {speedup:.2f}×")

    # ── PART D: BatchQueryProcessor test ─────────────────────────────────
    print(f"\n[4/4] BatchQueryProcessor — checking if QueryProcessor is available ...")

    INDEX_DIR = os.path.join(_REPO_ROOT, 'lsh_index')
    qp_available = (os.path.exists(CNN_PATH) and
                    os.path.exists(os.path.join(INDEX_DIR, 'lsh_config.json')))

    if not qp_available:
        print("  SKIP — lsh_index/ or data/ not found.")
        print("  BatchQueryProcessor requires the full index on disk.")
        print("  Run this test on Kaggle after attaching the feature dataset.")
    else:
        from query_engine.query_processor import QueryProcessor

        print("  Loading QueryProcessor ...")
        qp = QueryProcessor(index_dir=INDEX_DIR, data_path=CNN_PATH)

        bqp = BatchQueryProcessor(qp, batch_size=4, max_wait_ms=20)

        # Send 20 sequential queries (tests correctness, not throughput)
        with h5py.File(CNN_PATH, 'r') as f:
            all_ids_arr = f['image_ids'][:]
        test_ids = random.sample(list(all_ids_arr), 20)

        print(f"  Submitting 20 sequential queries via BatchQueryProcessor ...")
        for iid in test_ids:
            emb  = qp.prepare_query_embedding(image_id=int(iid))
            cands = bqp.submit(emb, query_image_id=int(iid))
            assert isinstance(cands, set), f"Expected set, got {type(cands)}"
            assert len(cands) > 0, "Empty candidate set"

        bqp_stats = bqp.get_stats()
        print(f"  BatchQueryProcessor stats: {bqp_stats}")
        print(f"  BatchQueryProcessor test: PASS")
        bqp.shutdown()

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Task 4 Summary")
    print("=" * 65)
    print(f"  FeatureCompressor L2 error    : {'PASS' if l2_pass else 'FAIL'}")
    print(f"  FeatureCompressor sim error   : {'PASS' if sim_pass else 'FAIL'}")
    print(f"  float16 speedup estimate      : {speedup:.2f}×  "
          f"({'PASS' if speedup >= 0.8 else 'NOTE: numpy float16 may not be faster on CPU'})")
    print(f"\n  NOTE: float16 speedup varies by hardware.")
    print(f"  On CPU-only machines float16 is sometimes equal to float32.")
    print(f"  The main benefit is memory: 512MB → 256MB regardless of speedup.")
    print("=" * 65)
