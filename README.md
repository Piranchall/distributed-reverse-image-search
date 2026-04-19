# Distributed Reverse Image Search Engine

A distributed reverse image search engine that identifies visually similar images across a 1-million-image corpus using parallel feature extraction, locality-sensitive hashing (LSH), and approximate nearest neighbor search.

---

## Project Structure

```
PDC_distributed-reverse-image-search/
│
├── feature_Extraction/          # Milestone 1 — CNN, hash, and descriptor extraction notebooks
├── feature_Fusion/              # Milestone 1 — Feature fusion module and similarity metrics
├── validation/                  # Milestone 1 — Ground truth dataset and evaluation framework
│
├── lsh_index/                   # Milestone 2 — LSH index structure and construction
│   ├── lsh_structure.py         # LSHConfig, projection matrices, ConsistentHashRing
│   ├── build_index.py           # Parallel index construction (multiprocessing)
│   ├── lsh_config.json          # Shared hyperparameters (all members read this)
│   └── hash_tables/             # 10 table_N.pkl + 10 table_N_balanced.pkl files
│
├── query_engine/                # Milestone 2 — Query processing and result ranking
│   ├── query_processor.py       # Parallel fan-out query across all LSH tables (threads)
│   ├── search_engine.py         # Full search pipeline with re-ranking via fusion.py
│   ├── test_queries.py          # Standalone query test script
│   └── search_engine_validation.ipynb  # Validation notebook (Kaggle)
│
├── load_balancer/               # Milestone 2 — Index sharding and load balancing
│   ├── load_balancer.py         # Hotspot detection, bucket splitting, benchmarking
│   ├── balance_report.json      # Per-table bucket size statistics
│   ├── split_buckets.json       # Keys of all split hotspot buckets
│   └── benchmark_results.json   # Latency comparison: balanced vs unbalanced
│
├── cache/                       # Milestone 2 — Query caching layer
│   ├── query_cache.py           # LRU cache with approximate Hamming-distance matching
│   ├── cache_stats.json         # Cache hit rate and latency statistics
│   └── cache_validation.ipynb   # Validation notebook (Kaggle)
│
└── data/                        # Large feature files — download separately (see below)
```

---

## Dataset

**MIRFLICKR-1M** — 1 million Flickr images used as the image corpus.
[kaggle.com/datasets/sohangundoju/mirflickr-1m](https://www.kaggle.com/datasets/sohangundoju/mirflickr-1m)

---

## Pre-extracted Feature Files

The merged feature files for all 1M images are too large to store in this repo. Download them separately and place in the `data/` folder.

**Full 1M feature files** (required for Tasks 3–6 full integration):
[kaggle.com/datasets/piranchalghai/pdc-micflickr-1m-features](https://www.kaggle.com/datasets/piranchalghai/pdc-micflickr-1m-features)

Files to download into `data/`:
```
all_cnn_embeddings.h5
all_perceptual_hashes.pkl
all_sift_features.pkl
all_orb_features.pkl
all_hist_features.pkl
all_image_paths.pkl
```

**Pre-built query engine dataset** (LSH index + feature files bundled together — use this to run Tasks 3–6 directly on Kaggle without re-building the index):
[Mahnoor's Kaggle dataset](https://www.kaggle.com/datasets/0eadbda4d48350616e810dad4f6369b1fc3f4033fa34388a82263c6566e7cd81)

---

## How to Run

### Milestone 2 — Build the LSH Index (Task 1 & 2)

Run from the repo root after downloading `all_cnn_embeddings.h5` into `data/`:

```bash
python lsh_index/lsh_structure.py   # generates lsh_config.json, projection_matrices.pkl, consistent_hash_ring.pkl
python lsh_index/build_index.py     # generates hash_tables/table_0.pkl … table_9.pkl
```

### Milestone 2 — Test the Query Engine (Task 3 & 4)

```bash
python query_engine/test_queries.py       # runs 20 random queries, prints candidates and latency
python query_engine/search_engine.py      # runs integration test against ground_truth.json
```

### Milestone 2 — Run Load Balancer (Task 5)

```bash
python load_balancer/load_balancer.py     # profiles index, splits hotspot buckets, writes benchmark
```

---

## System Overview

| Component | Description | PDC Technique |
|---|---|---|
| Feature extraction | CNN (ResNet-50, 128-dim PCA), pHash, SIFT/ORB/Histograms | GPU batch processing, parallel workers |
| LSH indexing | 10 independent hash tables, 12-bit keys, 4,096 buckets/table | Multiprocessing, consistent hashing |
| Query processing | Fan-out to all 10 tables simultaneously | ThreadPoolExecutor parallelism |
| Re-ranking | Full 5-feature weighted similarity (60% CNN + 20% hash + 10% SIFT + 5% ORB + 5% Histogram) | Vectorized NumPy batch similarity |
| Load balancing | Hotspot detection + 16-way bucket splitting | Index sharding |
| Caching | LRU cache with approximate pHash matching (Hamming ≤ 8 bits) | Cache-aside pattern |

---

## Key Results

- **Index built:** 1,000,000 images indexed across 10 tables in 13.5 seconds (12-core CPU)
- **Query latency:** Mean ~83ms unbalanced → ~56ms after load balancing
- **P99 latency:** 356ms → 202ms after balancing (43% improvement)
- **Cache hit rate:** 50% on repeated query workloads
- **Retrieval accuracy:** Recall@5 = 1.0 on validation set (all 5 known variants found in top 5)

---

## Requirements

```
torch
torchvision
opencv-python-headless
Pillow
imagehash
numpy
scipy
scikit-learn
h5py
tqdm
requests
matplotlib
pandas
```

Install with:
```bash
pip install torch torchvision opencv-python-headless Pillow imagehash numpy scipy scikit-learn h5py tqdm requests matplotlib pandas
```

