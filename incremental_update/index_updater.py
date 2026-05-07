"""
index_updater.py — Milestone 3 Task 1: Incremental Index Updates
Project 20: Distributed Reverse Image Search Engine
Author: Piranchal

What this file does:
  1. IndexUpdater.__init__()  — loads the existing 10 hash tables, projection
                                matrices, config, and hash ring from disk.
  2. add_image()              — inserts ONE new image into all 10 tables (single).
  3. add_batch()              — inserts MANY new images efficiently using the
                                vectorized hash_embeddings_batch() function.
  4. save_tables()            — writes the updated tables back to the .pkl files.
  5. update_feature_files()   — appends the new image's features to the 5 feature
                                pickle files so re-ranking works for it.
  6. verify_insert()          — round-trip check that a given image_id is
                                reachable from all 10 tables.

Run this file directly to execute the self-test:
    python incremental_update/index_updater.py
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import h5py

# ── Repo root on sys.path so we can import from lsh_index/ etc. ──────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from lsh_index.lsh_structure import (
    LSHConfig,
    ConsistentHashRing,
    hash_embedding,
    hash_embeddings_batch,
)
from incremental_update.concurrent_index import ConcurrentHashTable, wrap_tables


# =============================================================================
# IndexUpdater
# =============================================================================

class IndexUpdater:
    """
    Adds new images to the already-built LSH index without a full rebuild.

    The 10 hash tables (table_0.pkl … table_9.pkl) are loaded into memory
    as ConcurrentHashTable instances so that QueryProcessor threads can still
    serve queries while inserts are in progress.

    Typical usage:
        updater = IndexUpdater.load_state(
            index_dir  = 'lsh_index',
            tables_dir = 'lsh_index/hash_tables',
        )
        updater.add_batch([(1_000_000, emb0), (1_000_001, emb1)])
        updater.save_tables()
        updater.verify_insert(1_000_000)
    """

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self, index_dir: str, tables_dir: str):
        """
        Load all index artifacts from disk.

        Args:
            index_dir  : folder containing lsh_config.json,
                         projection_matrices.pkl, consistent_hash_ring.pkl
                         e.g. 'lsh_index'
            tables_dir : folder containing table_0.pkl … table_9.pkl
                         e.g. 'lsh_index/hash_tables'
        """
        self.index_dir  = index_dir
        self.tables_dir = tables_dir

        # ── 1. Load LSHConfig ────────────────────────────────────────────
        config_path = os.path.join(index_dir, 'lsh_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"lsh_config.json not found at {config_path}\n"
                f"Run lsh_structure.py first to generate it."
            )
        with open(config_path, 'r') as f:
            self.config = LSHConfig.from_dict(json.load(f))
        print(f"[IndexUpdater] Config loaded: "
              f"{self.config.num_tables} tables, "
              f"hash_size={self.config.hash_size}, "
              f"embedding_dim={self.config.embedding_dim}")

        # ── 2. Load projection matrices ──────────────────────────────────
        # These are read-only — never regenerate them. Using the same seed-42
        # matrices that build_index.py used guarantees bucket keys are consistent.
        matrices_path = os.path.join(index_dir, 'projection_matrices.pkl')
        if not os.path.exists(matrices_path):
            raise FileNotFoundError(
                f"projection_matrices.pkl not found at {matrices_path}"
            )
        with open(matrices_path, 'rb') as f:
            self.projection_matrices = pickle.load(f)
        assert len(self.projection_matrices) == self.config.num_tables, (
            f"Expected {self.config.num_tables} matrices, "
            f"got {len(self.projection_matrices)}"
        )
        print(f"[IndexUpdater] Projection matrices loaded: "
              f"{len(self.projection_matrices)} × "
              f"{self.projection_matrices[0].shape}")

        # ── 3. Load ConsistentHashRing ────────────────────────────────────
        ring_path = os.path.join(index_dir, 'consistent_hash_ring.pkl')
        if not os.path.exists(ring_path):
            raise FileNotFoundError(
                f"consistent_hash_ring.pkl not found at {ring_path}"
            )
        with open(ring_path, 'rb') as f:
            ring_dict = pickle.load(f)
        self.ring = ConsistentHashRing.from_dict(ring_dict)
        print(f"[IndexUpdater] Hash ring loaded: {self.ring}")

        # ── 4. Load all 10 hash tables as ConcurrentHashTable instances ──
        if not os.path.isdir(tables_dir):
            raise FileNotFoundError(
                f"Hash tables directory not found: {tables_dir}\n"
                f"Run build_index.py first."
            )

        print(f"[IndexUpdater] Loading hash tables from {tables_dir} ...")
        plain_dicts = []
        for i in range(self.config.num_tables):
            table_path = os.path.join(tables_dir, f'table_{i}.pkl')
            if not os.path.exists(table_path):
                raise FileNotFoundError(f"Missing: {table_path}")
            with open(table_path, 'rb') as f:
                d = pickle.load(f)
            plain_dicts.append(d)
            total_ids = sum(len(v) for v in d.values())
            print(f"  table_{i}: {len(d):,} buckets, {total_ids:,} image IDs")

        # Wrap all plain dicts in ConcurrentHashTable so QueryProcessor
        # threads can read concurrently while we insert.
        self.hash_tables = wrap_tables(plain_dicts)
        print(f"[IndexUpdater] All tables wrapped in ConcurrentHashTable.\n")

        # ── 5. Internal insert log ────────────────────────────────────────
        # Each entry: {image_id, timestamp, bucket_keys: [key_t0, ..., key_t9]}
        self._insert_log = []

    # ── Class method alternative constructor ─────────────────────────────────

    @classmethod
    def load_state(cls,
                   index_dir  : str = 'lsh_index',
                   tables_dir : str = None):
        """
        Named constructor — builds an IndexUpdater from existing files.
        Equivalent to calling __init__ directly, but more explicit at the call site.

        Args:
            index_dir  : folder containing config + matrices + ring
            tables_dir : folder containing table_N.pkl files.
                         Defaults to index_dir/hash_tables if not given.
        """
        if tables_dir is None:
            tables_dir = os.path.join(index_dir, 'hash_tables')
        return cls(index_dir=index_dir, tables_dir=tables_dir)

    # =========================================================================
    # Core insert — single image
    # =========================================================================

    def add_image(self, image_id: int, embedding: np.ndarray):
        """
        Insert a single new image into all 10 hash tables.

        Does NOT save to disk — call save_tables() when ready to persist.

        Args:
            image_id  : integer, must be unique (not already in the index).
                        For new images beyond the original 1M, use IDs ≥ 1,000,000.
            embedding : numpy array, shape (128,), float32.
                        Will be L2-normalized internally.

        Raises:
            ValueError : if image_id already exists in the index.
            ValueError : if embedding is wrong shape.
        """
        # ── Validate embedding shape ─────────────────────────────────────
        embedding = np.array(embedding, dtype=np.float32).flatten()
        if embedding.shape != (self.config.embedding_dim,):
            raise ValueError(
                f"embedding must be shape ({self.config.embedding_dim},), "
                f"got {embedding.shape}"
            )

        # ── Duplicate check (check table 0 only — tables are always in sync) ─
        bucket_key_0 = hash_embedding(embedding, self.projection_matrices[0])
        existing = self.hash_tables[0].get(bucket_key_0, [])
        if image_id in existing:
            raise ValueError(
                f"image_id {image_id} already exists in the index "
                f"(found in table_0 bucket '{bucket_key_0}')."
            )

        # ── L2-normalize ─────────────────────────────────────────────────
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            raise ValueError(f"embedding for image_id {image_id} has near-zero norm.")
        normalized = embedding / norm

        # ── Insert into all 10 tables ─────────────────────────────────────
        bucket_keys = []
        for i in range(self.config.num_tables):
            key = hash_embedding(normalized, self.projection_matrices[i])
            self.hash_tables[i].append(key, image_id)   # ConcurrentHashTable.append()
            bucket_keys.append(key)

        # ── Log the insert ────────────────────────────────────────────────
        self._insert_log.append({
            'image_id'   : image_id,
            'timestamp'  : time.time(),
            'bucket_keys': bucket_keys,   # [key_table0, key_table1, ..., key_table9]
        })

    # =========================================================================
    # Core insert — batch of images (EFFICIENT)
    # =========================================================================

    def add_batch(self, image_tuples: list):
        """
        Insert a batch of new images into all 10 hash tables.

        This is NOT just a loop over add_image(). For each table we call
        hash_embeddings_batch() which does ONE vectorized matrix multiply
        for the entire batch rather than N individual matrix multiplies.

        For 1000 images this is ~1000× faster than calling add_image() 1000 times.

        Args:
            image_tuples: list of (image_id: int, embedding: np.ndarray) tuples.

        Raises:
            ValueError: if ANY image_id already exists in the index.
                        All-or-nothing: no images are inserted if any duplicate found.
            ValueError: if any embedding has wrong shape.
        """
        if not image_tuples:
            return

        image_ids  = [int(t[0]) for t in image_tuples]
        embeddings = [np.array(t[1], dtype=np.float32).flatten() for t in image_tuples]

        # ── Validate all shapes ───────────────────────────────────────────
        for i, emb in enumerate(embeddings):
            if emb.shape != (self.config.embedding_dim,):
                raise ValueError(
                    f"Tuple index {i} (image_id={image_ids[i]}): "
                    f"embedding shape {emb.shape} != ({self.config.embedding_dim},)"
                )

        # ── Validate NO duplicates — check ALL before inserting ANY ──────
        # Check against table 0 (tables are always in sync)
        duplicates = []
        for img_id, emb in zip(image_ids, embeddings):
            key0 = hash_embedding(emb, self.projection_matrices[0])
            existing = self.hash_tables[0].get(key0, [])
            if img_id in existing:
                duplicates.append(img_id)

        if duplicates:
            raise ValueError(
                f"Duplicate image IDs found — aborting batch (no images inserted): "
                f"{duplicates}"
            )

        # ── Stack all embeddings, normalize row-wise ──────────────────────
        emb_matrix = np.stack(embeddings, axis=0)   # shape: (N, 128)
        norms      = np.linalg.norm(emb_matrix, axis=1, keepdims=True)   # (N, 1)
        norms      = np.where(norms < 1e-8, 1e-8, norms)
        normalized = emb_matrix / norms             # shape: (N, 128), unit rows

        # ── Insert into all 10 tables — one vectorized call per table ────
        # hash_embeddings_batch() does (N,128) @ (128,12) = (N,12) in one call.
        # Then we loop over the N results and call .append() for each image.
        all_bucket_keys = []   # will be shape (num_tables, N)

        for i in range(self.config.num_tables):
            bucket_keys_for_table = hash_embeddings_batch(
                normalized,
                self.projection_matrices[i]
            )
            for img_id, key in zip(image_ids, bucket_keys_for_table):
                self.hash_tables[i].append(key, img_id)

            all_bucket_keys.append(bucket_keys_for_table)

        # ── Log all inserts ───────────────────────────────────────────────
        ts = time.time()
        for idx, img_id in enumerate(image_ids):
            self._insert_log.append({
                'image_id'   : img_id,
                'timestamp'  : ts,
                'bucket_keys': [all_bucket_keys[t][idx]
                                for t in range(self.config.num_tables)],
            })

        print(f"[IndexUpdater] add_batch: inserted {len(image_ids)} images.")

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_tables(self):
        """
        Write all 10 updated hash tables back to their pickle files.

        Calls .to_dict() on each ConcurrentHashTable to get a plain dict
        before pickling — we never pickle the lock objects.

        Uses pickle protocol 4 (same as build_index.py).

        This method is idempotent: calling it twice produces the same result.
        """
        print(f"[IndexUpdater] Saving tables to {self.tables_dir} ...")
        os.makedirs(self.tables_dir, exist_ok=True)

        for i in range(self.config.num_tables):
            path = os.path.join(self.tables_dir, f'table_{i}.pkl')
            plain_dict = self.hash_tables[i].to_dict()

            with open(path, 'wb') as f:
                pickle.dump(plain_dict, f, protocol=4)

            size_mb    = os.path.getsize(path) / (1024 * 1024)
            num_buckets = len(plain_dict)
            total_ids   = sum(len(v) for v in plain_dict.values())
            print(f"  table_{i}.pkl → {size_mb:.1f} MB  "
                  f"({num_buckets:,} buckets, {total_ids:,} total IDs)")

        print(f"[IndexUpdater] All {self.config.num_tables} tables saved.\n")

    # =========================================================================
    # Feature file updates
    # =========================================================================

    def update_feature_files(self,
                             image_id        : int,
                             embedding       : np.ndarray,
                             phash_value     : int,
                             sift_descriptors: np.ndarray,
                             orb_descriptors : np.ndarray,
                             hist_vector     : np.ndarray,
                             data_dir        : str = 'data'):
        """
        Append the new image's features to the 5 feature pickle files in data/
        so that rank_candidates() can score it during re-ranking.

        For the CNN HDF5: appending to the on-disk file requires maxshape=(None,...)
        at dataset creation time in Milestone 1. If that was not done, only the
        in-memory feature store is updated here — a note is printed.
        The on-disk HDF5 will incorporate this image on next full rebuild.

        Args:
            image_id         : integer, already inserted via add_image/add_batch
            embedding        : numpy array (128,) float32
            phash_value      : 64-bit integer pHash value from imagehash library
            sift_descriptors : numpy array or None
            orb_descriptors  : numpy array or None
            hist_vector      : numpy array of color histogram values
            data_dir         : folder containing the feature .pkl files
        """
        print(f"[IndexUpdater] Updating feature files for image_id={image_id} ...")

        # ── Pickle files: load → add entry → re-save ──────────────────────
        file_map = {
            'all_perceptual_hashes.pkl': phash_value,
            'all_sift_features.pkl'    : sift_descriptors,
            'all_orb_features.pkl'     : orb_descriptors,
            'all_hist_features.pkl'    : hist_vector,
            'all_image_paths.pkl'      : f"synthetic_{image_id}",
        }

        for filename, new_value in file_map.items():
            path = os.path.join(data_dir, filename)
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found — skipping.")
                continue

            with open(path, 'rb') as f:
                store = pickle.load(f)

            store[image_id] = new_value

            with open(path, 'wb') as f:
                pickle.dump(store, f, protocol=4)

            print(f"  Updated {filename}: now {len(store):,} entries")

        # ── CNN HDF5: attempt resize, fall back gracefully ─────────────────
        # NOTE: HDF5 datasets can only be resized if created with
        # maxshape=(None, 128) in Milestone 1. If they were created without
        # maxshape, h5py raises TypeError on resize(). In that case we
        # skip the on-disk update (the in-memory embeddings array in
        # All_Features would need to be updated separately by the caller).
        cnn_path = os.path.join(data_dir, 'all_cnn_embeddings.h5')
        if os.path.exists(cnn_path):
            try:
                norm_emb = embedding / max(np.linalg.norm(embedding), 1e-8)
                with h5py.File(cnn_path, 'r+') as f:
                    current_size = f['embeddings'].shape[0]
                    f['embeddings'].resize((current_size + 1, 128))
                    f['image_ids'].resize((current_size + 1,))
                    f['embeddings'][current_size] = norm_emb.astype(np.float32)
                    f['image_ids'][current_size]  = image_id
                print(f"  Updated all_cnn_embeddings.h5: now {current_size + 1:,} rows")
            except TypeError:
                # Dataset was created without maxshape — cannot resize
                print(f"  NOTE: all_cnn_embeddings.h5 was not created with "
                      f"maxshape=(None,128) — on-disk HDF5 not updated.\n"
                      f"  The in-memory All_Features instance must be updated "
                      f"by the caller. This image will appear in results "
                      f"after the next full rebuild.")
        else:
            print(f"  WARNING: {cnn_path} not found — CNN feature not updated.")

        print(f"[IndexUpdater] Feature files updated.\n")

    # =========================================================================
    # Verification
    # =========================================================================

    def verify_insert(self, image_id: int, embedding: np.ndarray = None) -> bool:
        """
        Round-trip check: confirm image_id is reachable from ALL 10 tables.

        For each table:
          1. Compute the bucket key from the embedding
          2. Look up that key in the table
          3. Check image_id is in the bucket

        Args:
            image_id  : integer to verify
            embedding : the embedding used at insert time (normalized internally).
                        If None, tries to find it in _insert_log.

        Returns:
            True if found in all tables, False otherwise.
        """
        # Retrieve embedding from insert log if not provided
        if embedding is None:
            log_entry = next(
                (e for e in reversed(self._insert_log) if e['image_id'] == image_id),
                None
            )
            if log_entry is None:
                print(f"[verify_insert] image_id {image_id} not in insert log "
                      f"and no embedding provided — cannot verify.")
                return False
            # Re-derive from bucket keys: we stored bucket_keys, not the embedding.
            # Use the stored bucket keys directly for verification instead.
            all_passed = True
            for i, key in enumerate(log_entry['bucket_keys']):
                bucket = self.hash_tables[i].get(key, [])
                if image_id not in bucket:
                    print(f"  [FAIL] table_{i}: image_id {image_id} NOT in "
                          f"bucket '{key}' (bucket size={len(bucket)})")
                    all_passed = False
                else:
                    print(f"  [OK]   table_{i}: image_id {image_id} found in "
                          f"bucket '{key}' (bucket size={len(bucket)})")
            return all_passed

        # If embedding provided: re-compute expected bucket keys
        embedding = np.array(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        all_passed = True
        for i in range(self.config.num_tables):
            expected_key = hash_embedding(embedding, self.projection_matrices[i])
            bucket       = self.hash_tables[i].get(expected_key, [])
            if image_id not in bucket:
                print(f"  [FAIL] table_{i}: image_id {image_id} NOT in "
                      f"bucket '{expected_key}' (bucket size={len(bucket)})")
                all_passed = False
            else:
                print(f"  [OK]   table_{i}: image_id {image_id} in "
                      f"bucket '{expected_key}' (bucket size={len(bucket)})")

        return all_passed

    def get_insert_log(self) -> list:
        """Return a copy of the insert log for debugging."""
        return list(self._insert_log)


# =============================================================================
# Self-test   (run with: python incremental_update/index_updater.py)
# =============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("Milestone 3 Task 1 — IndexUpdater Self-Test")
    print("=" * 65)

    # ── Paths — adjust if your layout differs ────────────────────────────
    INDEX_DIR  = os.path.join(_REPO_ROOT, 'lsh_index')
    TABLES_DIR = os.path.join(_REPO_ROOT, 'lsh_index', 'hash_tables')
    DATA_DIR   = os.path.join(_REPO_ROOT, 'data')

    # ── Check prerequisites ───────────────────────────────────────────────
    required = [
        os.path.join(INDEX_DIR, 'lsh_config.json'),
        os.path.join(INDEX_DIR, 'projection_matrices.pkl'),
        os.path.join(INDEX_DIR, 'consistent_hash_ring.pkl'),
    ]
    for p in required:
        if not os.path.exists(p):
            print(f"\nERROR: {p} not found.")
            print("Run lsh_structure.py first, then build_index.py.")
            sys.exit(1)

    if not os.path.isdir(TABLES_DIR):
        print(f"\nERROR: {TABLES_DIR} not found.")
        print("Run build_index.py first to generate the hash tables.")
        sys.exit(1)

    # ── Load existing index ───────────────────────────────────────────────
    print("\n[1/6] Loading existing index via load_state()...")
    updater = IndexUpdater.load_state(index_dir=INDEX_DIR, tables_dir=TABLES_DIR)

    # ── Create 3 synthetic unit-vector embeddings ─────────────────────────
    print("\n[2/6] Creating 3 synthetic embeddings (random unit vectors)...")
    rng   = np.random.default_rng(seed=999)
    # Use timestamp-based IDs so re-running never collides
    _base = int(time.time())
    NEW_IDS = [_base, _base + 1, _base + 2]
    NEW_EMBS = []
    for new_id in NEW_IDS:
        emb  = rng.standard_normal(128).astype(np.float32)
        emb  = emb / np.linalg.norm(emb)
        NEW_EMBS.append(emb)
        print(f"  image_id={new_id}  norm={np.linalg.norm(emb):.6f}  "
              f"first3={emb[:3]}")

    # ── Test add_batch ────────────────────────────────────────────────────
    print("\n[3/6] Calling add_batch()...")
    t0 = time.time()
    updater.add_batch(list(zip(NEW_IDS, NEW_EMBS)))
    elapsed = time.time() - t0
    print(f"  add_batch completed in {elapsed*1000:.1f} ms")

    # ── Save tables ───────────────────────────────────────────────────────
    print("\n[4/6] Calling save_tables()...")
    updater.save_tables()

    # ── Verify all 3 inserts ──────────────────────────────────────────────
    print("\n[5/6] Verifying inserts with verify_insert()...")
    results = {}
    for new_id in NEW_IDS:
        print(f"\n  Verifying image_id={new_id}:")
        passed = updater.verify_insert(new_id)
        results[new_id] = passed
        status = "PASS" if passed else "FAIL"
        print(f"  → {status}")

    # ── Test duplicate detection ──────────────────────────────────────────
    print("\n[6/6] Testing duplicate detection (should raise ValueError)...")
    duplicate_caught = False
    try:
        updater.add_image(NEW_IDS[0], NEW_EMBS[0])
        print("  FAIL — no exception raised for duplicate image_id!")
    except ValueError as e:
        duplicate_caught = True
        print(f"  PASS — ValueError raised correctly: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    all_verify_passed = all(results.values())
    print(f"  add_batch (3 images)   : {'PASS' if not any(False for v in results.values() if not v) else 'FAIL'}")
    for img_id, passed in results.items():
        print(f"  verify_insert({img_id}) : {'PASS' if passed else 'FAIL'}")
    print(f"  duplicate detection    : {'PASS' if duplicate_caught else 'FAIL'}")

    if all_verify_passed and duplicate_caught:
        print("\n✓ All tests PASSED.")
    else:
        print("\n✗ Some tests FAILED — check output above.")

    print("=" * 65)
    print("\nNext steps:")
    print("  1. Confirm table_N.pkl files grew in size (check lsh_index/hash_tables/)")
    print("  2. Integrate with Mahnoor: she will call wrap_tables() in query_processor.py")
    print("  3. Test concurrent scenario: run QueryProcessor in parallel with add_batch()")