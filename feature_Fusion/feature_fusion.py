# Feature Fusion Module — Milestone 1 Phase B
# ───────────────────────────────────────────
# Loads all 6 feature files and provides:
#   - All_Features: loads and accesses all features by global image ID
#   - compute_similarity(): full weighted similarity between two images
#   - compute_batch_similarity(): fast vectorized CNN similarity for re-ranking
#   - rank_candidates(): returns top-k most similar images for a query
# ─────────────────────────────────────────────────────────────────────────────
import os
import pickle
import numpy as np
import h5py

#  Fusion weights — must sum to 1.0 
W_CNN  = 0.60
W_HASH = 0.20
W_SIFT = 0.10
W_ORB  = 0.05
W_HIST = 0.05

#  Hash pre-filter threshold 
# If average Hamming distance across all 3 hash types exceeds this, return 0.0 immediately without computing any other features.
HASH_THRESHOLD = 25   # out of 64 bits


# ──────── FEATURE STORE ────────
class All_Features:
    """
    Loads all 6 feature files into memory on initialization.
    Provides O(1) lookup for any feature type by global image ID.
    """

    def __init__(self,
                 cnn_path   : str, # cnn_path   : path to all_cnn_embeddings.h5
                 hash_path  : str, # hash_path  : path to all_perceptual_hashes.pkl
                 sift_path  : str, # sift_path  : path to all_sift_features.pkl
                 orb_path   : str, # orb_path   : path to all_orb_features.pkl
                 hist_path  : str, # hist_path  : path to all_hist_features.pkl
                 paths_path : str):# paths_path : path to all_image_paths.pkl
        
        print("Loading All_Features ...")

        #  CNN embeddings 
        print("  Loading CNN embeddings ...")
        with h5py.File(cnn_path, 'r') as f:
            self.embeddings = f['embeddings'][:]   # (N, 128) float32
            image_ids       = f['image_ids'][:]    # (N,) int64

        # Build O(1) lookup: global_id → row index in self.embeddings
        self.id_to_index = {int(gid): idx for idx, gid in enumerate(image_ids)}
        print(f"  ✓ CNN: {self.embeddings.shape[0]:,} embeddings, "
              f"dim={self.embeddings.shape[1]}")

        #  Perceptual hashes 
        print("  Loading perceptual hashes ...")
        with open(hash_path, 'rb') as f:
            self.hashes = pickle.load(f)
        print(f"  ✓ Hashes: {len(self.hashes):,} entries")

        #  Traditional features — loaded separately 
        print("  Loading SIFT features ...")
        with open(sift_path, 'rb') as f:
            self.sift = pickle.load(f)
        print(f"  ✓ SIFT: {len(self.sift):,} entries")

        print("  Loading ORB features ...")
        with open(orb_path, 'rb') as f:
            self.orb = pickle.load(f)
        print(f"  ✓ ORB: {len(self.orb):,} entries")

        print("  Loading histograms ...")
        with open(hist_path, 'rb') as f:
            self.hist = pickle.load(f)
        print(f"  ✓ Histograms: {len(self.hist):,} entries")

        #  Image path mapping 
        print("  Loading image paths ...")
        with open(paths_path, 'rb') as f:
            self.image_paths = pickle.load(f)
        print(f"  ✓ Image paths: {len(self.image_paths):,} entries")

        print("✓ All_Features ready.\n")

    # ──────── Accessors ────────
    def get_cnn(self, global_id: int) -> np.ndarray:
        """Returns 128-dim CNN embedding for a given image ID."""
        return self.embeddings[self.id_to_index[global_id]]

    def get_hashes(self, global_id: int) -> dict:
        """Returns {ahash, dhash, phash} integers for a given image ID."""
        return self.hashes[global_id]

    def get_sift(self, global_id: int) -> np.ndarray:
        """Returns 128-dim SIFT vector for a given image ID."""
        return self.sift[global_id]

    def get_orb(self, global_id: int) -> np.ndarray:
        """Returns 32-dim ORB vector for a given image ID."""
        return self.orb[global_id]

    def get_hist(self, global_id: int) -> np.ndarray:
        """Returns 96-dim color histogram for a given image ID."""
        return self.hist[global_id]

    def get_path(self, global_id: int) -> str:
        """Returns original file path for a given image ID."""
        return self.image_paths.get(global_id, f"unknown_id_{global_id}")


# ──────── SIMILARITY METRICS ────────
def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Normalizes both to unit length then takes dot product.
    Returns float in [0, 1] for typical image feature vectors.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(v1 / n1, v2 / n2))


def _hamming_distance(h1: int, h2: int) -> int:
    """Number of bit positions that differ between two hash integers."""
    return bin(h1 ^ h2).count('1')


def _hash_similarity(hashes1: dict, hashes2: dict) -> tuple:
    """
    Returns (avg_hamming_distance, similarity_score).
    avg_hamming is used for the pre-filter check. similarity_score is in [0, 1].
    """
    d_a = _hamming_distance(hashes1['ahash'], hashes2['ahash'])
    d_d = _hamming_distance(hashes1['dhash'], hashes2['dhash'])
    d_p = _hamming_distance(hashes1['phash'], hashes2['phash'])
    avg_dist = (d_a + d_d + d_p) / 3.0
    similarity = 1.0 - (avg_dist / 64.0)
    return avg_dist, similarity


def _histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Histogram intersection similarity.
    Both histograms are normalized so result is in [0, 1].
    """
    return float(np.sum(np.minimum(h1, h2)))



# ──────── TWO-STAGE FUSION ────────
def compute_similarity(id1: int, id2: int, fs: All_Features) -> float:
    """
    Computes weighted similarity between two images using all 5 feature types.

    Stage 1 — Fast hash pre-filter:
        If average Hamming distance > HASH_THRESHOLD, return 0.0 immediately. This skips ~95% of obviously dissimilar pairs.

    Stage 2 — Full weighted score:
        final_score = W_CNN * cnn_sim + W_HASH * hash_sim + W_SIFT * sift_sim + W_ORB * orb_sim + W_HIST * hist_sim

    Parameters:
    id1, id2 = global image IDs picked at random
    fs = All_Features instance

    Returns: float in [0, 1]
    """
    # Stage 1 — hash pre-filter (cheap)
    avg_hamming, hash_sim = _hash_similarity(
        fs.get_hashes(id1),
        fs.get_hashes(id2)
    )
    if avg_hamming > HASH_THRESHOLD:
        return 0.0

    # Stage 2 — full weighted score (expensive, only for similar candidates)
    cnn_sim  = _cosine_similarity(fs.get_cnn(id1),  fs.get_cnn(id2))
    sift_sim = _cosine_similarity(fs.get_sift(id1), fs.get_sift(id2))
    orb_sim  = _cosine_similarity(fs.get_orb(id1),  fs.get_orb(id2))
    hist_sim = _histogram_intersection(fs.get_hist(id1), fs.get_hist(id2))

    final_score = (W_CNN  * cnn_sim  +
                   W_HASH * hash_sim +
                   W_SIFT * sift_sim +
                   W_ORB  * orb_sim  +
                   W_HIST * hist_sim)

    return float(final_score)


# ──────── VECTORIZED BATCH SIMILARITY (CNN only — used during re-ranking) ────────
def compute_batch_similarity(query_id: int,
                              candidate_ids: list,
                              fs: All_Features) -> dict:
    """
    Computes CNN cosine similarity only (full fusion is too slow for hundreds of candidates) 
    between one query and many candidates in a single vectorized matrix operation.

    Parameters:
    query_id = global ID of the query image
    candidate_ids = list of global IDs to compare against
    fs =: All_Features instance

    Returns: dict {candidate_id: similarity_score} sorted by score descending
    """
    if not candidate_ids:
        return {}

    # Get query embedding and normalize
    query_vec = fs.get_cnn(query_id).astype(np.float32)
    q_norm    = np.linalg.norm(query_vec)
    if q_norm < 1e-8:
        return {cid: 0.0 for cid in candidate_ids}
    query_vec = query_vec / q_norm

    # Stack all candidate embeddings into matrix (N, 128)
    candidate_matrix = np.stack(
        [fs.get_cnn(cid).astype(np.float32) for cid in candidate_ids],
        axis=0
    )

    # Normalize each row to unit length
    norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)   # avoid division by zero
    candidate_matrix = candidate_matrix / norms

    # Compute all cosine similarities in one matrix multiply
    scores = candidate_matrix @ query_vec   # shape (N,)

    # Build result dict sorted by score descending
    result = {cid: float(scores[i]) for i, cid in enumerate(candidate_ids)}
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    return result

# ──────── RANK CANDIDATES ────────
def rank_candidates(query_id: int,
                    candidate_ids: list,
                    fs: All_Features,
                    top_k: int = 10) -> list:
    """
    Returns the top-k most similar images to a query from a candidate list. Uses vectorized CNN similarity for speed.

    Parameters:
    query_id = global ID of the query image
    candidate_ids = list of candidate global IDs returned by LSH
    fs = All_Features instance
    top_k =: number of results to return

    Returns: list of tuples [(candidate_id, score), ...] length top_k, best first
    """
    if not candidate_ids:
        return []

    scores = compute_batch_similarity(query_id, candidate_ids, fs)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]



# Testing if everything works as intended
if __name__ == '__main__':
    import random
    DATA_DIR = 'C:\\Users\\piran\\Downloads\\pdcproject'

    fs = All_Features(
        cnn_path   = os.path.join(DATA_DIR, 'all_cnn_embeddings.h5'),
        hash_path  = os.path.join(DATA_DIR, 'all_perceptual_hashes.pkl'),
        sift_path  = os.path.join(DATA_DIR, 'all_sift_features.pkl'),
        orb_path   = os.path.join(DATA_DIR, 'all_orb_features.pkl'),
        hist_path  = os.path.join(DATA_DIR, 'all_hist_features.pkl'),
        paths_path = os.path.join(DATA_DIR, 'all_image_paths.pkl'),
    )

    # Pick 2 random IDs
    all_ids    = list(fs.id_to_index.keys())
    id1, id2   = random.sample(all_ids, 2)

    print(f"Testing compute_similarity on IDs {id1} and {id2} ...")
    score = compute_similarity(id1, id2, fs)
    print(f"  Similarity score : {score:.4f}")
    print(f"  Image 1 path     : {fs.get_path(id1)}")
    print(f"  Image 2 path     : {fs.get_path(id2)}")

    print(f"\nTesting rank_candidates ...")
    query_id       = id1
    candidate_ids  = random.sample(all_ids, 10000)
    results        = rank_candidates(query_id, candidate_ids, fs, top_k=5)

    print(f"  Query ID : {query_id}")
    print(f"  Top 5 results:")
    for rank, (cid, s) in enumerate(results, 1):
        print(f"    {rank}. ID={cid}  score={s:.4f}  path={fs.get_path(cid)}")

    print("\n✓ Feature Fusion working correctly.")