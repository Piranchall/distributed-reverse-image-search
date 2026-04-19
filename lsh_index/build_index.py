"""
build_index.py — Task 2: Parallel Index Construction
Project 20: Distributed Reverse Image Search Engine — Milestone 2
Author: Piranchal

What this file does:
  1. Loads all 1,000,000 CNN embeddings from all_cnn_embeddings.h5 into RAM
  2. Hashes every embedding into all 10 LSH tables in parallel using multiprocessing
  3. Saves 10 pickle files: lsh_index/hash_tables/table_0.pkl … table_9.pkl

These 10 files are the complete search index. Mahnoor loads them in Task 3.

Run this file from the REPO ROOT after lsh_structure.py has been run:
    python lsh_index/build_index.py

Expected runtime: 10–30 minutes depending on CPU core count.
Expected output:  10 files, ~40–70 MB each, ~500–700 MB total.
"""

import os
import sys
import json
import pickle
import time
import multiprocessing
import numpy as np
import h5py
from tqdm import tqdm

# ── Allow import of lsh_structure from the same folder ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lsh_structure import (
    LSHConfig,
    hash_embeddings_batch,
    hash_embedding,
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — adjust if your folder layout differs
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = 'data/all_cnn_embeddings.h5'        # Milestone 1 merged output
CONFIG_PATH    = 'lsh_index/lsh_config.json'
MATRICES_PATH  = 'lsh_index/projection_matrices.pkl'
OUTPUT_DIR     = 'lsh_index/hash_tables'
BATCH_SIZE     = 10_000   # images processed per worker call (5 MB per batch)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings(data_path):
    """
    Load the full embeddings and image_ids arrays from the HDF5 file into RAM.

    WHY load into RAM fully:
        Random access into an HDF5 file is slow. Slicing numpy arrays in RAM
        for batching is ~100x faster than slicing from disk.

    Returns:
        embeddings: numpy array, shape (1000000, 128), float32
        image_ids:  numpy array, shape (1000000,),     int64
    """
    print(f"Loading embeddings from {data_path} ...")
    t0 = time.time()

    with h5py.File(data_path, 'r') as f:
        embeddings = f['embeddings'][:]    # loads entire dataset into RAM
        image_ids  = f['image_ids'][:]

    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  embeddings shape : {embeddings.shape}  dtype={embeddings.dtype}")
    print(f"  image_ids  shape : {image_ids.shape}  dtype={image_ids.dtype}")

    # ── Sanity checks ─────────────────────────────────────────────────────
    assert embeddings.shape == (1_000_000, 128), (
        f"Expected (1000000, 128), got {embeddings.shape}\n"
        f"Has the merge step from Milestone 1 been completed?"
    )
    assert image_ids.shape == (1_000_000,), (
        f"Expected (1000000,), got {image_ids.shape}"
    )
    assert int(image_ids[0]) == 0, (
        f"Expected image_ids[0] == 0, got {image_ids[0]}"
    )
    assert int(image_ids[-1]) == 999_999, (
        f"Expected image_ids[-1] == 999999, got {image_ids[-1]}"
    )

    print("  All shape and ID assertions PASSED")
    return embeddings, image_ids


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Worker Function (MUST be top-level for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def process_batch(args):
    """
    Worker function: hash one batch of embeddings into all 10 tables.

    IMPORTANT: This function MUST stay at the top level of the module.
    Python's multiprocessing.Pool pickles worker functions to send to child
    processes. Nested functions (defined inside another function/class) cannot
    be pickled and will raise a PicklingError.

    Args:
        args: tuple of (batch_embeddings, batch_image_ids, projection_matrices)
              batch_embeddings:  numpy array, shape (BATCH_SIZE, 128)
              batch_image_ids:   numpy array, shape (BATCH_SIZE,)
              projection_matrices: list of 10 numpy arrays, each (12, 128)

    Returns:
        List of 10 sublists. Each sublist contains (image_id_int, bucket_key_str)
        tuples for every image in this batch, for that table index.

        result[0] = [(42, '101100011010'), (43, '110010001101'), ...]  # table 0
        result[1] = [(42, '001011010110'), (43, '100101100010'), ...]  # table 1
        ...
    """
    batch_embeddings, batch_image_ids, projection_matrices = args

    result = []
    for matrix in projection_matrices:
        # Hash the entire batch at once (vectorized, much faster than looping)
        bucket_keys = hash_embeddings_batch(batch_embeddings, matrix)

        # Pair each image_id with its bucket key
        table_entries = [
            (int(img_id), bucket_key)
            for img_id, bucket_key in zip(batch_image_ids, bucket_keys)
        ]
        result.append(table_entries)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Build the Index
# ─────────────────────────────────────────────────────────────────────────────

def build_index(embeddings, image_ids, projection_matrices, num_workers=None):
    """
    Hash all 1M embeddings into 10 LSH tables using parallel workers.

    Strategy:
        - Divide embeddings into batches of BATCH_SIZE (10,000 each = 100 batches)
        - Each worker receives ONE batch (5 MB) — not the full 500 MB array
        - Workers run in parallel via multiprocessing.Pool
        - Main process accumulates results into 10 dictionaries

    Args:
        embeddings:          numpy array (1000000, 128)
        image_ids:           numpy array (1000000,)
        projection_matrices: list of 10 numpy arrays, each (12, 128)
        num_workers:         number of parallel processes (default: all CPU cores)

    Returns:
        hash_tables: list of 10 dicts, each {bucket_key_str: [image_id, ...]}
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    num_images = len(embeddings)
    num_tables = len(projection_matrices)
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nBuilding index:")
    print(f"  Images:      {num_images:,}")
    print(f"  Tables:      {num_tables}")
    print(f"  Batch size:  {BATCH_SIZE:,}")
    print(f"  Batches:     {num_batches}")
    print(f"  Workers:     {num_workers}")

    # ── Prepare all batch argument tuples ────────────────────────────────
    # Each tuple contains only the slice for that batch — NOT the full array.
    # This is critical for memory safety: passing the full array to each
    # worker would copy 500 MB × num_workers worth of data.
    print("\nPreparing batches...")
    batch_args = []
    for start in range(0, num_images, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_images)
        batch_emb = embeddings[start:end]      # slice: only BATCH_SIZE rows
        batch_ids = image_ids[start:end]
        batch_args.append((batch_emb, batch_ids, projection_matrices))

    print(f"  {len(batch_args)} batches prepared")

    # ── Initialize empty hash tables ─────────────────────────────────────
    hash_tables = [{} for _ in range(num_tables)]

    # ── Run parallel processing ───────────────────────────────────────────
    print(f"\nRunning parallel hashing with {num_workers} workers...")
    t0 = time.time()

    with multiprocessing.Pool(processes=num_workers) as pool:
        # imap_unordered yields results as workers finish (not in batch order)
        # This keeps the progress bar responsive and memory usage low
        for batch_result in tqdm(
            pool.imap_unordered(process_batch, batch_args),
            total=num_batches,
            desc="Hashing batches",
            unit="batch"
        ):
            # batch_result is a list of 10 sublists (one per table)
            for table_idx, table_entries in enumerate(batch_result):
                table_dict = hash_tables[table_idx]
                for img_id, bucket_key in table_entries:
                    # Append this image_id to its bucket
                    if bucket_key in table_dict:
                        table_dict[bucket_key].append(img_id)
                    else:
                        table_dict[bucket_key] = [img_id]

    elapsed = time.time() - t0
    print(f"\nParallel hashing complete in {elapsed:.1f}s  "
          f"({elapsed/60:.1f} minutes)")

    return hash_tables


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(hash_tables):
    """
    Print per-table statistics to verify index quality.

    Expected values with HASH_SIZE=12 and 1M images:
        Buckets per table:   ~2,000–4,096 (out of 4,096 possible)
        Assignments/table:   exactly 1,000,000
        Mean bucket size:    ~244 images
        Max bucket size:     typically < 2,000 (flag if > 10,000)
    """
    print(f"\n{'=' * 65}")
    print(f"{'Table':<10} {'Buckets':>8} {'Total IDs':>10} "
          f"{'Min':>6} {'Mean':>7} {'P95':>7} {'Max':>7}")
    print('-' * 65)

    for i, table in enumerate(hash_tables):
        bucket_sizes = [len(v) for v in table.values()]
        total_ids    = sum(bucket_sizes)
        num_buckets  = len(table)
        min_size     = min(bucket_sizes)
        mean_size    = np.mean(bucket_sizes)
        p95_size     = int(np.percentile(bucket_sizes, 95))
        max_size     = max(bucket_sizes)

        # Flag tables with extreme imbalance
        flag = " ⚠️" if max_size > 10_000 else ""

        print(f"  table_{i:<5} {num_buckets:>8,} {total_ids:>10,} "
              f"{min_size:>6} {mean_size:>7.1f} {p95_size:>7} {max_size:>7}{flag}")

        # Critical check: every image must be indexed exactly once per table
        assert total_ids == 1_000_000, (
            f"table_{i} has {total_ids} assignments — expected 1,000,000!\n"
            f"Some images are missing from this table."
        )

    print('=' * 65)
    print("All tables have exactly 1,000,000 assignments — PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Save Tables
# ─────────────────────────────────────────────────────────────────────────────

def save_tables(hash_tables, output_dir):
    """
    Save each hash table as a separate pickle file.

    Format: table_N.pkl → Python dict {bucket_key_str: [image_id_int, ...]}

    This is the exact format Mahnoor's QueryProcessor (Task 3) will load.

    Args:
        hash_tables: list of 10 dicts from build_index()
        output_dir:  directory path, e.g. 'lsh_index/hash_tables'
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving {len(hash_tables)} table files to {output_dir}/")
    total_bytes = 0

    for i, table in enumerate(hash_tables):
        filename = os.path.join(output_dir, f'table_{i}.pkl')

        with open(filename, 'wb') as f:
            pickle.dump(table, f, protocol=4)

        size_mb = os.path.getsize(filename) / (1024 * 1024)
        total_bytes += os.path.getsize(filename)
        print(f"  table_{i}.pkl  → {size_mb:.1f} MB  "
              f"({len(table):,} buckets)")

    total_mb = total_bytes / (1024 * 1024)
    print(f"\nTotal index size: {total_mb:.0f} MB across {len(hash_tables)} files")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Correctness Test
# ─────────────────────────────────────────────────────────────────────────────

def run_correctness_test(hash_tables, embeddings, image_ids, projection_matrices):
    """
    Verify that image_id 42 can be found in table_0 by hashing its embedding.

    This checks the full round-trip:
        embedding → hash_embedding() → bucket_key → table_0[bucket_key] → image_id

    Also runs an informal similarity test: images in the same LSH bucket
    should have higher average cosine similarity than random images.
    """
    print("\nRunning correctness test...")

    # ── Round-trip test for image ID 42 ──────────────────────────────────
    target_id  = 42
    target_idx = int(np.where(image_ids == target_id)[0][0])
    target_emb = embeddings[target_idx]

    # Hash it into table 0
    bucket_key = hash_embedding(target_emb, projection_matrices[0])
    bucket     = hash_tables[0].get(bucket_key, [])

    if target_id in bucket:
        print(f"  Round-trip test PASSED: image_id {target_id} found in "
              f"table_0['{bucket_key}'] ({len(bucket)} images in bucket)")
    else:
        print(f"  Round-trip test FAILED: image_id {target_id} NOT in "
              f"table_0['{bucket_key}']. Bucket contents[:10]: {bucket[:10]}")
        return

    # ── Similarity test ───────────────────────────────────────────────────
    # Pick up to 5 random images from the SAME bucket
    same_bucket_ids = [img_id for img_id in bucket if img_id != target_id][:5]

    # Pick 5 random images from a DIFFERENT bucket
    other_keys = [k for k in hash_tables[0].keys() if k != bucket_key]
    import random
    random.shuffle(other_keys)
    diff_bucket_ids = []
    for k in other_keys:
        diff_bucket_ids.extend(hash_tables[0][k][:2])
        if len(diff_bucket_ids) >= 5:
            break
    diff_bucket_ids = diff_bucket_ids[:5]

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # Compute average similarities
    same_sims = []
    for img_id in same_bucket_ids:
        idx = int(np.where(image_ids == img_id)[0][0])
        same_sims.append(cosine_sim(target_emb, embeddings[idx]))

    diff_sims = []
    for img_id in diff_bucket_ids:
        idx = int(np.where(image_ids == img_id)[0][0])
        diff_sims.append(cosine_sim(target_emb, embeddings[idx]))

    avg_same = np.mean(same_sims) if same_sims else 0
    avg_diff = np.mean(diff_sims) if diff_sims else 0

    print(f"\n  Similarity test for image_id {target_id} in table_0:")
    print(f"    Avg cosine sim with same-bucket images:  {avg_same:.4f}")
    print(f"    Avg cosine sim with diff-bucket images:  {avg_diff:.4f}")

    if avg_same > avg_diff:
        print("  LSH quality check PASSED — same-bucket images are more similar")
    else:
        print("  LSH quality check WARNING — expected same-bucket to be more similar")
        print("  This can happen by chance with very small samples — not necessarily a bug")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Windows requires this guard for multiprocessing to work correctly
    multiprocessing.freeze_support()

    print("=" * 65)
    print("Task 2: Parallel Index Construction")
    print("=" * 65)

    # ── Step 1: Check Task 1 outputs exist ───────────────────────────────
    for path in [CONFIG_PATH, MATRICES_PATH]:
        if not os.path.exists(path):
            print(f"\nERROR: {path} not found.")
            print("Run lsh_structure.py first to generate Task 1 artifacts.")
            sys.exit(1)

    # ── Step 2: Load Task 1 artifacts ────────────────────────────────────
    print("\nLoading Task 1 artifacts...")

    with open(CONFIG_PATH) as f:
        config = LSHConfig.from_dict(json.load(f))
    print(f"  Config loaded: {config.num_tables} tables, hash_size={config.hash_size}")

    with open(MATRICES_PATH, 'rb') as f:
        projection_matrices = pickle.load(f)
    print(f"  Projection matrices loaded: {len(projection_matrices)} matrices, "
          f"shape={projection_matrices[0].shape}")

    # ── Step 3: Load embeddings ───────────────────────────────────────────
    embeddings, image_ids = load_embeddings(DATA_PATH)

    # ── Step 4: Build the index ───────────────────────────────────────────
    hash_tables = build_index(
        embeddings,
        image_ids,
        projection_matrices,
        num_workers=os.cpu_count()
    )

    # ── Step 5: Verify statistics ─────────────────────────────────────────
    print_stats(hash_tables)

    # ── Step 6: Correctness test ──────────────────────────────────────────
    run_correctness_test(hash_tables, embeddings, image_ids, projection_matrices)

    # ── Step 7: Save all tables ───────────────────────────────────────────
    save_tables(hash_tables, OUTPUT_DIR)

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("Task 2 COMPLETE. Files produced:")
    for i in range(config.num_tables):
        print(f"  lsh_index/hash_tables/table_{i}.pkl")
    print(f"\nNEXT ACTIONS:")
    print(f"  1. Upload ALL files in lsh_index/ to PDC_Project_Shared/lsh_index/ on Drive")
    print(f"  2. Announce in team chat that the index is ready")
    print(f"  3. Mahnoor, Hayatullah, and Aliza can now download and start their tasks")
    print('=' * 65)
