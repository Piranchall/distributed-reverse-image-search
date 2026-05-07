# Distributed Reverse Image Search Engine

A distributed reverse image search engine that identifies visually similar images across a corpus of one million photographs using parallel feature extraction, locality-sensitive hashing (LSH), and approximate nearest neighbour search. The system accepts any query image and returns the top-K visually similar results in under 10 milliseconds.

**Dataset:** MIRFLICKR-1M — 1 million Flickr photographs  
**Team:** Piranchal (Leader) · Mahnoor Mughal · Aliza Muhammad Warris Poonjani · Hayatullah  
**Course:** Parallel and Distributed Computing — Project 20

---

## Key Results

| Metric | Value |
|---|---|
| Images indexed | 1,000,000 (MIRFLICKR-1M) |
| Index build time | 13.5 seconds (12-worker multiprocessing) |
| Mean query latency (balanced) | 56 ms |
| P99 latency (balanced) | 202 ms (43% improvement over unbalanced) |
| P99 latency — 1M corpus, 12 workers (M3) | 3.5 ms (LSH lookup only) |
| P50 / P99 latency at 10M scale | 6.2 ms / 27.2 ms |
| Recall@5 on validation set | 1.00 (M2) · 0.876 baseline (M3) |
| Best M3 config recall@5 | 0.976 (num_tables=15, hash_size=10) |
| Incremental insert speed | 1–3 ms for new image (vs 13,500 ms full rebuild) |
| Cache hit speedup | 8× faster than full search on hit |

---


## Dataset and Feature Files

### Image Dataset

**MIRFLICKR-1M** is used as the primary corpus — 1 million Flickr photographs in 10 packs of 100,000 each.  
→ [kaggle.com/datasets/sohangundoju/mirflickr-1m](https://www.kaggle.com/datasets/sohangundoju/mirflickr-1m)

### Pre-extracted Feature Files

Feature extraction for 1M images took ~25 hours total on Kaggle T4 GPUs. Download the pre-extracted files instead of re-running extraction.

**Full 1M feature files** — place in `data/`:  
→ [kaggle.com/datasets/piranchalghai/pdc-micflickr-1m-features](https://www.kaggle.com/datasets/piranchalghai/pdc-micflickr-1m-features)

```
data/
├── all_cnn_embeddings.h5        # 1M × 128-dim PCA-compressed CNN embeddings (~512 MB)
├── all_perceptual_hashes.pkl    # 1M × {ahash, dhash, phash} integers (~80 MB)
├── all_sift_features.pkl        # 1M × 128-dim mean-pooled SIFT vectors (~200 MB)
├── all_orb_features.pkl         # 1M × 32-dim mean-pooled ORB vectors (~80 MB)
├── all_hist_features.pkl        # 1M × 96-dim colour histograms (~150 MB)
└── all_image_paths.pkl          # 1M × {global_id: file_path} (~50 MB)
```

**Pre-built Kaggle dataset** (LSH index + feature files bundled — use this to run query engine and M3 benchmarks directly on Kaggle without rebuilding):  
→ [Mahnoor's Kaggle dataset](https://www.kaggle.com/datasets/mahnoormughal16/pdc-ml3)

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision opencv-python-headless Pillow imagehash numpy scipy scikit-learn h5py tqdm requests matplotlib pandas
```

Clone the repo and download the feature files into `data/` before running any of the steps below.

---

### Milestone 1 — Feature Extraction (Kaggle only)

Feature extraction requires Kaggle T4 GPU access and the MIRFLICKR-1M dataset attached as a Kaggle input. Run the three extraction notebooks once per pack (10 packs × 3 notebooks = 30 runs), then run the merge script.

```
feature_Extraction/CNN_Embeddings.ipynb           # ResNet-50 inference, PCA compression
feature_Extraction/hashes.ipynb                   # pHash / dHash / aHash
feature_Extraction/Traditional_Descriptors.ipynb  # SIFT, ORB, colour histograms
```

If you have downloaded the pre-extracted files from Kaggle, skip this step entirely.

---

### Milestone 2 — Build the LSH Index

Run from the repo root after placing the feature files in `data/`:

```bash
# Step 1: Generate LSH structure (config, projection matrices, consistent hash ring)
python lsh_index/lsh_structure.py

# Step 2: Build 10 hash tables from 1M CNN embeddings (takes ~14 seconds, 12 workers)
python lsh_index/build_index.py

# Step 3: Run load balancer — splits 1,608 hotspot buckets
python load_balancer/load_balancer.py
```

Expected output after Step 2:
```
Images: 1,000,000 | Tables: 10 | Batch: 10,000 | Workers: 12
Parallel hashing complete in 13.8s
All tables have exactly 1,000,000 assignments — PASSED
Round-trip test: image_id 42 found in table_0 — PASSED
```

### Milestone 2 — Run the Query Engine

```bash
# Run 20 random queries and print candidates + latency
python query_engine/test_queries.py

# Run integration test against ground_truth.json
python query_engine/search_engine.py
```

---

### Milestone 3 — Incremental Updates (Tasks 1 & 2)

```bash
# Runs full self-test: adds 3 synthetic images, saves tables, verifies inserts
python incremental_update/index_updater.py

# Runs concurrency stress tests for ReadersWriterLock and ConcurrentHashTable
python incremental_update/concurrent_index.py
```

Expected output:
```
[IndexUpdater] add_batch: inserted 3 images.  (1–3 ms)
All tables: 1,000,003 total IDs — PASSED
verify_insert: 30 / 30 checks passed — PASSED
Duplicate detection: PASS
```

---

### Milestone 3 — Parallel Re-ranking and Compression (Tasks 3 & 4)

Tasks 3 and 4 are integrated into `search_engine.py` and validated via the Kaggle notebook:

```
query_engine/milestone3_tasks3_4_validation.ipynb   # run on Kaggle with full 1M data
```

To toggle float16 compression, set the flag at the top of `search_engine.py`:

```python
USE_COMPRESSION = True   # halves embedding memory: 512 MB → 256 MB
```

---

### Milestone 3 — Scaling Benchmarks (Task 5)

```bash
# Builds 100K sub-sampled index, runs strong & weak scaling experiments
# Saves scaling_results.json and 3 PNG plots to benchmarks/
python benchmarks/scaling_benchmark.py
```

The 10M index is only built if disk space > 20 GB is available (Kaggle). Locally the script runs 100K and 1M benchmarks only and skips 10M gracefully.

---

### Milestone 3 — Accuracy Benchmarks (Task 6)

```bash
# Tests 10 LSH configurations on the 600-image test set
# Saves accuracy_results.json and 2 PNG plots to benchmarks/
python benchmarks/accuracy_benchmark.py
```

No large feature files required — uses only the 600 test images in `validation/test_features/`.

---

## System Architecture

```
Query Image
    │
    ▼
ResNet-50 + PCA (Milestone 1)
    │  128-dim unit-vector embedding
    ▼
Cache Lookup — LRU with pHash Hamming matching (Milestone 2)
    │  cache miss
    ▼
LSH Fan-out — ThreadPoolExecutor across 10 tables (Milestone 2)
    │  ~500–5,000 candidate IDs
    ▼
Parallel Re-ranker — vectorised CNN scoring per shard (Milestone 3)
    │  top-K sorted results
    ▼
Return (image_id, similarity_score) list
```

New images can be inserted into the live index without rebuilding via `IndexUpdater.add_batch()` while concurrent queries continue safely through `ConcurrentHashTable`.

---

## PDC Concepts Implemented

| Concept | Where | Detail |
|---|---|---|
| GPU batch processing | M1 CNN extraction | ResNet-50 inference at 64 images/pass, ~700 img/s |
| Pipeline parallelism | M1 extraction | CNN, hash, SIFT/ORB ran as independent pipelines per pack |
| CPU parallelism (threads) | M1 hashing | ThreadPoolExecutor — I/O-bound, GIL released by PIL |
| CPU parallelism (processes) | M1 SIFT/ORB | multiprocessing.Pool — CPU-bound, bypasses GIL |
| Distributed indexing | M2 LSH | 10 independent tables with consistent hash ring |
| Parallel index construction | M2 build | multiprocessing.Pool, imap_unordered, 12 workers |
| Parallel query processing | M2 query | ThreadPoolExecutor fan-out to all 10 tables |
| Consistent hashing | M2 structure | MD5-based ring, 30 virtual positions, uniform distribution |
| Load balancing | M2 balancer | Hotspot detection + 16-way MD5 bucket splitting |
| Cache-aside pattern | M2 cache | LRU + approximate Hamming-distance matching |
| Concurrent data structures | M3 Task 2 | ReadersWriterLock — many readers, exclusive writer |
| Incremental computation | M3 Task 1 | add_batch(): 1–3 ms vs 13,500 ms full rebuild |
| Task parallelism | M3 Task 3 | Candidate set sharded across persistent ThreadPoolExecutor |
| Data compression | M3 Task 4 | float32 → float16: 512 MB → 256 MB, error < 0.001 |
| Scaling analysis | M3 Task 5 | Strong scaling (workers) + weak scaling (corpus + workers) |

---

## Feature Weights

The five-feature similarity metric used in re-ranking:

| Feature | Weight | Dimension | Role |
|---|---|---|---|
| CNN embedding (ResNet-50 + PCA) | 60% | 128 | Semantic visual similarity |
| Perceptual hash (aHash + dHash + pHash) | 20% | 3 × 64-bit | Near-duplicate detection |
| SIFT descriptors | 10% | 128 | Keypoint structural similarity |
| ORB descriptors | 5% | 32 | Binary feature matching |
| Colour histogram | 5% | 96 | Spatial colour distribution |

---

## LSH Configuration

All hyperparameters are stored in `lsh_index/lsh_config.json` and loaded by every component. Nothing is hardcoded.

| Parameter | Value | Meaning |
|---|---|---|
| num_tables | 10 | Independent hash tables |
| hash_size | 12 | Bits per key → 4,096 possible buckets per table |
| embedding_dim | 128 | CNN embedding dimension (fixed by M1 PCA) |
| random_seed | 42 | Reproducible projection matrices |
| max_candidates | 5,000 | Maximum candidates before re-ranking |
| top_k | 10 | Default results per query |
| cache_max_size | 1,000 | Maximum LRU cache entries |
| cache_hamming_threshold | 8 | Hamming distance for approximate cache hit |

**Recommended upgrade (from M3 Task 6 analysis):** Setting `num_tables=15` and `hash_size=10` increases Recall@5 from 0.876 to 0.976 at identical mean query latency.

---

## Validation Results

### Milestone 1 — Feature Extraction

| k | Precision@k | Recall@k |
|---|---|---|
| 1 | 1.00 | 0.20 |
| 5 | 1.00 | 1.00 |
| 10 | 0.50 | 1.00 |

Targets: Precision@1 > 0.70 ✓ · Recall@10 > 0.80 ✓

### Milestone 3 — Accuracy vs Latency (top 5 configurations)

| Config | Recall@5 | Precision@10 | Mean Latency |
|---|---|---|---|
| t15_h10 | 0.976 | 0.488 | 0.11 ms |
| t10_h8  | 0.968 | 0.484 | 0.11 ms |
| t20_h12 | 0.956 | 0.478 | 0.15 ms |
| t10_h10 | 0.936 | 0.468 | 0.08 ms |
| **t10_h12 (baseline)** | **0.876** | **0.438** | **0.10 ms** |

### Milestone 3 — Scaling Benchmarks

**Strong scaling (1M corpus):**

| Workers | P50 | P95 | P99 |
|---|---|---|---|
| 1  | 2.0 ms | 4.0 ms | 8.6 ms |
| 2  | 2.0 ms | 3.0 ms | 4.0 ms |
| 12 | 2.0 ms | 3.0 ms | 3.5 ms |

**Weak scaling:**

| Corpus | Workers | P50 | P99 |
|---|---|---|---|
| 100K | 1  | 1.0 ms | 1.3 ms |
| 1M   | 10 | 2.0 ms | 3.7 ms |
| 10M  | 12 | 6.2 ms | 27.2 ms |

---

## Project Structure

```
distributed-reverse-image-search/
│
├── feature_Extraction/                    # Milestone 1
│   ├── CNN_Embeddings.ipynb               # ResNet-50 + PCA extraction (Kaggle, GPU)
│   ├── hashes.ipynb                       # pHash / dHash / aHash extraction
│   └── Traditional_Descriptors.ipynb     # SIFT / ORB / colour histogram extraction
│
├── feature_Fusion/                        # Milestone 1
│   └── feature_fusion.py                 # All_Features store + 5-feature weighted similarity
│
├── validation/                            # Milestone 1
│   ├── evaluator.py                       # precision_at_k(), recall_at_k()
│   ├── ground_truth.json                  # 100 base queries, each with 5 known variants
│   ├── generate_test_dataset.ipynb        # 600-image augmented test set generation
│   └── test_features/                     # Pre-extracted features for 600 test images
│       ├── test_cnn.h5
│       ├── test_hashes.pkl
│       ├── test_sift.pkl
│       ├── test_orb.pkl
│       ├── test_hist.pkl
│       └── test_image_paths.pkl
│
├── lsh_index/                             # Milestone 2
│   ├── lsh_structure.py                   # LSHConfig, projection matrices, ConsistentHashRing
│   ├── build_index.py                     # Parallel index construction (multiprocessing)
│   ├── lsh_config.json                    # Shared hyperparameters (all components read this)
│   ├── projection_matrices.pkl            # 10 × (12, 128) Gaussian projection matrices (seed 42)
│   ├── consistent_hash_ring.pkl           # 30-position consistent hash ring
│   └── hash_tables/                       # 10 × table_N.pkl + 10 × table_N_balanced.pkl
│
├── query_engine/                          # Milestone 2 + Milestone 3 patches
│   ├── query_processor.py                 # ThreadPoolExecutor fan-out to all 10 tables
│   ├── search_engine.py                   # Full pipeline: LSH → ParallelReranker → top-K
│   ├── test_queries.py                    # Standalone query test script
│   ├── search_engine_validation.ipynb     # M2 validation notebook (Kaggle)
│   └── milestone3_tasks3_4_validation.ipynb  # M3 Tasks 3 & 4 validation (Kaggle)
│
├── load_balancer/                         # Milestone 2
│   ├── load_balancer.py                   # Hotspot detection, 16-way bucket splitting
│   ├── balance_report.json                # Per-table bucket size statistics
│   ├── split_buckets.json                 # Keys of all 1,608 split hotspot buckets
│   └── benchmark_results.json            # Latency comparison: balanced vs unbalanced
│
├── cache/                                 # Milestone 2
│   ├── query_cache.py                     # LRU cache with approximate Hamming-distance matching
│   ├── cache_stats.json                   # Cache hit rate and latency statistics
│   └── cache_validation.ipynb            # Validation notebook (Kaggle)
│
├── incremental_update/                    # Milestone 3 — Tasks 1 & 2 (Piranchal)
│   ├── __init__.py
│   ├── index_updater.py                   # Add new images to live index (add_image, add_batch)
│   └── concurrent_index.py               # ReadersWriterLock + ConcurrentHashTable + wrap_tables()
│
├── reranking/                             # Milestone 3 — Task 3 (Mahnoor)
│   ├── __init__.py
│   └── parallel_reranker.py              # Parallel candidate scoring with persistent ThreadPoolExecutor
│
├── compression/                           # Milestone 3 — Task 4 (Mahnoor)
│   ├── __init__.py
│   └── feature_compressor.py             # float32→float16, CompressedEmbeddingStore, BatchQueryProcessor
│
├── benchmarks/                            # Milestone 3 — Tasks 5 & 6
│   ├── __init__.py
│   ├── scaling_benchmark.py              # P50/P95/P99 at 100K/1M/10M, strong & weak scaling (Hayatullah)
│   ├── accuracy_benchmark.py             # precision@k, recall@k vs latency across LSH configs (Aliza)
│   ├── scaling_results.json              # Strong and weak scaling numerical results
│   ├── accuracy_results.json             # All 10 configuration accuracy + latency measurements
│   ├── strong_scaling_latency.png
│   ├── strong_scaling_speedup.png
│   ├── weak_scaling_latency.png
│   ├── accuracy_latency_tradeoff.png
│   ├── parameter_sensitivity.png
│   └── indices/
│       └── 100k/                          # Sub-sampled ~100K index (table_N_100k.pkl × 10)
│
└── data/                                  # Large feature files — download separately (see below)
```

> **Note:** `benchmarks/indices/10m/` is not committed to this repo (10 × 48 MB = 480 MB). The 10M index is generated on Kaggle by running `scaling_benchmark.py`.

---