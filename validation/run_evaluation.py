# =============================================================================
# run_evaluation.py
# Evaluates the feature extraction and fusion system using the test dataset.
#
# Run order:
#   1. generate_test_dataset.py   → creates 600 test images + ground_truth.json
#   2. Extract features for test images (CNN, hashing, SIFT notebooks)
#   3. THIS FILE                  → measures Precision@k and Recall@k
#
# Expected results if system is working well:
#   Precision@1  > 0.70
#   Recall@10    > 0.80
# =============================================================================

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Base directory — root of your cloned repo
BASE_DIR = r'C:\\Users\\piran\\Downloads\\distributed-reverse-image-search'

# Add both folders to path so imports work
sys.path.append(os.path.join(BASE_DIR, 'feature_Fusion'))
sys.path.append(os.path.join(BASE_DIR, 'validation'))

from evaluator import evaluate_single_query
from feature_fusion import All_Features, rank_candidates

# ── Paths ───
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, 'validation', 'ground_truth.json')
RESULTS_IMG_PATH  = os.path.join(BASE_DIR, 'validation', 'evaluation_results.png')
RESULTS_CSV_PATH  = os.path.join(BASE_DIR, 'validation', 'evaluation_results.csv')

# Test features are inside validation/test_features/
TEST_FEATURES_DIR = os.path.join(BASE_DIR, 'validation', 'test_features')

CNN_PATH   = os.path.join(TEST_FEATURES_DIR, 'test_cnn.h5')
HASH_PATH  = os.path.join(TEST_FEATURES_DIR, 'test_hashes.pkl')
SIFT_PATH  = os.path.join(TEST_FEATURES_DIR, 'test_sift.pkl')
ORB_PATH   = os.path.join(TEST_FEATURES_DIR, 'test_orb.pkl')
HIST_PATH  = os.path.join(TEST_FEATURES_DIR, 'test_hist.pkl')
PATHS_PATH = os.path.join(TEST_FEATURES_DIR, 'test_image_paths.pkl')

# ── Verify all files exist before starting ────────────────────────────────────
print("=" * 50)
print("EVALUATION PIPELINE")
print("=" * 50)

missing = []
for name, path in [
    ('ground_truth.json', GROUND_TRUTH_PATH),
    ('test_cnn.h5',       CNN_PATH),
    ('test_hashes.pkl',   HASH_PATH),
    ('test_sift.pkl',     SIFT_PATH),
    ('test_orb.pkl',      ORB_PATH),
    ('test_hist.pkl',     HIST_PATH),
    ('test_image_paths',  PATHS_PATH),
]:
    if not os.path.exists(path):
        missing.append(name)

if missing:
    print(f"\n✗ Missing files: {missing}")
    print("Run generate_test_dataset.py and extract features first.")
    sys.exit(1)

print("✓ All required files found.")

# ── Load ground truth ─────────────────────────────────────────────────────────
with open(GROUND_TRUTH_PATH, 'r') as f:
    ground_truth = json.load(f)

# Convert string keys to integers (JSON keys are always strings)
ground_truth = {int(k): v for k, v in ground_truth.items()}

print(f"\nGround truth loaded")
print(f"  Base images : {len(ground_truth)}")
print(f"  Per image   : 5 variants")
print(f"  Total       : {len(ground_truth) * 6} test images")

# Sample check
print(f"\nSample ground truth entries:")
for base_id, variants in list(ground_truth.items())[:3]:
    print(f"  Base {base_id} → variants {variants}")

# ── Build full list of all test image IDs ─────────────────────────────────────
all_test_ids = []
for base_id, variant_ids in ground_truth.items():
    all_test_ids.append(base_id)
    all_test_ids.extend(variant_ids)

print(f"\nTotal candidate pool: {len(all_test_ids)} images")

# ── Load All_Features with TEST features ──────────────────────────────────────
# Note: these are the features extracted from the 600 test images only.
# NOT the 1M main dataset features.
print("\nLoading test feature store ...")
fs = All_Features(
    cnn_path   = CNN_PATH,
    hash_path  = HASH_PATH,
    sift_path  = SIFT_PATH,
    orb_path   = ORB_PATH,
    hist_path  = HIST_PATH,
    paths_path = PATHS_PATH
)

# ── Run evaluation ────────────────────────────────────────────────────────────
k_values          = [1, 5, 10, 20]
precision_results = {k: [] for k in k_values}
recall_results    = {k: [] for k in k_values}

print(f"\nRunning evaluation on {len(ground_truth)} base image queries ...")
print("-" * 50)

for idx, query_id in enumerate(ground_truth.keys()):

    # Real search using rank_candidates from fusion.py
    # Searches across all 600 test images (pool includes base + all variants)
    # Exclude the query image itself from candidates
    candidates   = [cid for cid in all_test_ids if cid != query_id]
    ranked       = rank_candidates(query_id, candidates, fs, top_k=20)
    retrieved_ids = [cid for cid, score in ranked]

    # Evaluate
    eval_results = evaluate_single_query(
        query_id, retrieved_ids, ground_truth, k_values
    )

    for k in k_values:
        precision_results[k].append(eval_results[k]["precision"])
        recall_results[k].append(eval_results[k]["recall"])

    if (idx + 1) % 10 == 0:
        print(f"  Done {idx + 1}/{len(ground_truth)} queries")

# ── Compute averages ──────────────────────────────────────────────────────────
mean_precision = {k: np.mean(precision_results[k]) for k in k_values}
mean_recall    = {k: np.mean(recall_results[k])    for k in k_values}

# ── Print results table ───────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"{'k':<8} {'Precision@k':<16} {'Recall@k':<16}")
print("-" * 50)
for k in k_values:
    p = mean_precision[k]
    r = mean_recall[k]
    p_status = "✓" if p >= 0.70 else "✗"
    r_status = "✓" if r >= 0.80 else "✗"
    print(f"{k:<8} {p:<14.4f} {p_status}   {r:<14.4f} {r_status}")

print("\nTarget: Precision@1 > 0.70, Recall@10 > 0.80")

# ── Save results to CSV ───────────────────────────────────────────────────────
import csv
os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
with open(RESULTS_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['k', 'precision_at_k', 'recall_at_k'])
    for k in k_values:
        writer.writerow([k, mean_precision[k], mean_recall[k]])

print(f"\n✓ Results saved to {RESULTS_CSV_PATH}")

# ── Plot results ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))

plt.plot(k_values, [mean_precision[k] for k in k_values],
         'b-o', label='Precision@k', linewidth=2, markersize=8)
plt.plot(k_values, [mean_recall[k] for k in k_values],
         'r-o', label='Recall@k', linewidth=2, markersize=8)

# Target lines
plt.axhline(y=0.70, color='blue',  linestyle='--', alpha=0.4, label='Precision target (0.70)')
plt.axhline(y=0.80, color='red',   linestyle='--', alpha=0.4, label='Recall target (0.80)')

plt.xlabel('k (number of top results)', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Retrieval Quality — Precision and Recall at k', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.xticks(k_values)

plt.tight_layout()
plt.savefig(RESULTS_IMG_PATH, dpi=150)
print(f"✓ Graph saved to {RESULTS_IMG_PATH}")
plt.show()

# ── Interpretation ────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("INTERPRETATION")
print("=" * 50)

p1  = mean_precision[1]
r10 = mean_recall[10]

if p1 >= 0.70 and r10 >= 0.80:
    print("✓ System meets quality targets.")
    print("  Feature extraction and fusion are working correctly.")
    print("  Ready to proceed to Milestone 2.")
elif p1 >= 0.50:
    print("~ System is partially working.")
    print("  Results are above random but below target.")
    print("  Consider tuning fusion weights in fusion.py.")
else:
    print("✗ System below expected quality.")
    print("  Possible issues:")
    print("  - Feature extraction may have errors")
    print("  - Test image IDs may not match feature file IDs")
    print("  - fusion.py hash threshold may be too aggressive")
    print("  Check that test features were extracted correctly.")

print("\n" + "=" * 50)
print("Evaluation Complete!")
print("=" * 50)