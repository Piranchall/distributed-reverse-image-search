"""
lsh_structure.py — Task 1: LSH Structure and Consistent Hashing
Project 20: Distributed Reverse Image Search Engine — Milestone 2
Author: Piranchal

What this file does:
  1. Defines LSHConfig — shared hyperparameters used by ALL team members
  2. Generates random projection matrices for hashing
  3. Defines hash_embedding() and hash_embeddings_batch()
  4. Implements ConsistentHashRing for distributed bucket assignment
  5. Saves lsh_config.json, projection_matrices.pkl, consistent_hash_ring.pkl

Run this file directly to generate all three output artifacts:
    python lsh_index/lsh_structure.py
"""

import os
import sys
import json
import pickle
import hashlib
import bisect
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Configuration
# ─────────────────────────────────────────────────────────────────────────────

class LSHConfig:
    """
    Central hyperparameter store shared by all team members.

    Mahnoor, Hayatullah, and Aliza all load this with:
        config = LSHConfig.from_dict(json.load(open('lsh_index/lsh_config.json')))

    NEVER hardcode these numbers in your own code — always read from the config
    object so a single change here propagates everywhere.
    """

    def __init__(self):
        # ── Index structure ──────────────────────────────────────────────
        self.num_tables           = 10    # number of independent LSH hash tables
        self.hash_size            = 12    # bits per hash → 2^12 = 4096 buckets/table
        self.embedding_dim        = 128   # CNN embedding size (fixed by Milestone 1 PCA)
        self.random_seed          = 42    # RNG seed — MUST be the same for everyone

        # ── Query behaviour ──────────────────────────────────────────────
        self.top_k                = 10    # default results returned per query
        self.max_candidates       = 5000  # max candidate IDs before re-ranking

        # ── Cache (Aliza Task 6) ─────────────────────────────────────────
        self.cache_max_size       = 1000  # max entries in LRU cache
        self.cache_hamming_threshold = 8  # perceptual hash bit-distance for cache hit

    def to_dict(self):
        """Return plain Python dict — safe to write with json.dump()."""
        return {
            "num_tables":              int(self.num_tables),
            "hash_size":               int(self.hash_size),
            "embedding_dim":           int(self.embedding_dim),
            "random_seed":             int(self.random_seed),
            "top_k":                   int(self.top_k),
            "max_candidates":          int(self.max_candidates),
            "cache_max_size":          int(self.cache_max_size),
            "cache_hamming_threshold": int(self.cache_hamming_threshold),
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct an LSHConfig from a dict loaded from lsh_config.json."""
        cfg = cls()
        cfg.num_tables              = int(d["num_tables"])
        cfg.hash_size               = int(d["hash_size"])
        cfg.embedding_dim           = int(d["embedding_dim"])
        cfg.random_seed             = int(d["random_seed"])
        cfg.top_k                   = int(d["top_k"])
        cfg.max_candidates          = int(d["max_candidates"])
        cfg.cache_max_size          = int(d["cache_max_size"])
        cfg.cache_hamming_threshold = int(d["cache_hamming_threshold"])
        return cfg

    def __repr__(self):
        return f"LSHConfig({self.to_dict()})"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Projection Matrices
# ─────────────────────────────────────────────────────────────────────────────

def generate_projection_matrices(config):
    """
    Generate NUM_TABLES random projection matrices.

    Each matrix has shape (HASH_SIZE, EMBEDDING_DIM) = (12, 128).
    Values are sampled from a standard normal distribution (mean=0, std=1).

    WHY normal distribution:
        Gaussian random projections preserve pairwise distances by the
        Johnson-Lindenstrauss lemma. Uniform projections do not have this.

    WHY seeded RNG:
        Seed 42 ensures every team member who runs this gets byte-identical
        matrices. If matrices differ between runs, hash keys differ and the
        index is incompatible with the query engine.

    Returns:
        List of config.num_tables numpy arrays, each shape (12, 128), dtype float32.
    """
    rng = np.random.default_rng(config.random_seed)

    matrices = []
    for i in range(config.num_tables):
        matrix = rng.standard_normal(
            size=(config.hash_size, config.embedding_dim)
        ).astype(np.float32)
        matrices.append(matrix)

    return matrices


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Hash Functions
# ─────────────────────────────────────────────────────────────────────────────

def hash_embedding(embedding, projection_matrix):
    """
    Hash a single 128-dim embedding into a bucket key string.

    How it works:
        1. Compute dot products: (12, 128) @ (128,) → (12,) vector
        2. Each dot product's sign becomes a bit:  >= 0 → '1',  < 0 → '0'
        3. Join 12 bits into a string like '101100011010'

    Args:
        embedding:         numpy array, shape (128,), float32
        projection_matrix: numpy array, shape (12, 128), float32

    Returns:
        12-character string of '0' and '1' — the bucket key for this table.
    """
    dot_products = projection_matrix @ embedding          # shape: (12,)
    bits = (dot_products >= 0).astype(np.int8)            # 1 where positive, 0 where negative
    return ''.join(str(b) for b in bits)


def hash_embeddings_batch(embeddings_matrix, projection_matrix):
    """
    Vectorized batch version of hash_embedding() — used in Task 2 for speed.

    MUST produce identical results to hash_embedding() for every individual row.

    How it works:
        1. Compute dot products: (N, 128) @ (128, 12) → (N, 12) matrix
        2. Apply sign comparison across the whole matrix at once
        3. Convert each row to a string

    Args:
        embeddings_matrix: numpy array, shape (N, 128), float32
        projection_matrix: numpy array, shape (12, 128), float32

    Returns:
        List of N strings, each 12 characters of '0'/'1'.
    """
    # (N, 128) @ (128, 12) → (N, 12)
    dot_products = embeddings_matrix @ projection_matrix.T  # shape: (N, 12)

    # Apply sign comparison: True/False → '1'/'0'
    bits_matrix = (dot_products >= 0).astype(np.int8)       # shape: (N, 12)

    # Convert each row to a string
    keys = []
    for row in bits_matrix:
        keys.append(''.join(str(b) for b in row))

    return keys


def verify_hash_consistency(embedding, projection_matrix):
    """
    Sanity check: single and batch versions must produce identical output.
    Raises AssertionError if they differ.
    """
    single_key = hash_embedding(embedding, projection_matrix)
    batch_key  = hash_embeddings_batch(embedding.reshape(1, -1), projection_matrix)[0]

    assert single_key == batch_key, (
        f"MISMATCH — single: '{single_key}', batch: '{batch_key}'\n"
        f"This indicates a broadcasting bug in hash_embeddings_batch()."
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Consistent Hashing Ring
# ─────────────────────────────────────────────────────────────────────────────

class ConsistentHashRing:
    """
    A consistent hashing ring that maps bucket keys to logical node (table file) names.

    In this project, "nodes" are the 10 pickle files: table_0.pkl … table_9.pkl.
    The ring assigns each possible bucket key to exactly one table file.

    WHY consistent hashing:
        When nodes are added/removed, only the buckets on the affected ring arc
        need to move. All other assignments stay the same. This is essential for
        incremental index updates in Milestone 3.

    WHY virtual nodes:
        Without virtual nodes, 10 real nodes on a random ring would create
        very unequal arc lengths → some tables would hold far more buckets.
        With 3 virtual nodes per real node (30 ring positions total), the
        distribution is much more uniform.

    Internal state:
        self._ring — sorted list of (ring_position_int, node_id_string) tuples
    """

    def __init__(self):
        self._ring = []   # sorted list of (position, node_id)

    def add_node(self, node_id, num_virtual_nodes=3):
        """
        Add a node to the ring with num_virtual_nodes virtual copies.

        For each virtual copy i, we hash the string f'{node_id}_virtual_{i}'
        with MD5, take the first 4 bytes as an unsigned integer, and insert
        at that position on the ring.

        Args:
            node_id:           string, e.g. 'table_0'
            num_virtual_nodes: number of virtual copies (default 3)
        """
        for i in range(num_virtual_nodes):
            key_bytes = f'{node_id}_virtual_{i}'.encode('utf-8')
            digest    = hashlib.md5(key_bytes).digest()
            position  = int.from_bytes(digest[:4], byteorder='big')

            # Insert in sorted order using bisect
            entry = (position, node_id)
            index = bisect.bisect_left(self._ring, entry)
            self._ring.insert(index, entry)

    def get_node(self, key_string):
        """
        Find which node is responsible for a given bucket key string.

        Process:
            1. Hash the bucket key string with MD5 → ring position
            2. Binary search for the first ring position >= query position
            3. Wrap around to index 0 if the query position is past all nodes

        Args:
            key_string: 12-character '0'/'1' bucket key from hash_embedding()

        Returns:
            node_id string, e.g. 'table_3'
        """
        if not self._ring:
            raise RuntimeError("Ring is empty — call add_node() first.")

        key_bytes = key_string.encode('utf-8')
        digest    = hashlib.md5(key_bytes).digest()
        position  = int.from_bytes(digest[:4], byteorder='big')

        # Find insertion point in the sorted ring
        # bisect_left gives us the first index where position could be inserted
        index = bisect.bisect_left(self._ring, (position, ''))

        # Wrap around to 0 if past the end
        if index >= len(self._ring):
            index = 0

        return self._ring[index][1]   # return the node_id

    def get_all_nodes(self):
        """Return sorted list of unique node IDs currently in the ring."""
        return sorted(set(node_id for _, node_id in self._ring))

    def to_dict(self):
        """
        Serialize ring state to a JSON-serializable dict.
        Format: {"ring": [[position_int, node_id_str], ...]}
        """
        return {
            "ring": [[int(pos), str(nid)] for pos, nid in self._ring]
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct a ConsistentHashRing from the dict produced by to_dict()."""
        ring = cls()
        ring._ring = [(int(pos), str(nid)) for pos, nid in d["ring"]]
        return ring

    def __len__(self):
        return len(self._ring)

    def __repr__(self):
        nodes = self.get_all_nodes()
        return f"ConsistentHashRing(nodes={nodes}, virtual_positions={len(self._ring)})"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Ring Initialization
# ─────────────────────────────────────────────────────────────────────────────

def build_ring(config):
    """
    Create a ConsistentHashRing and add all NUM_TABLES table nodes to it.

    With 10 tables and 3 virtual nodes each = 30 positions on the ring.

    Args:
        config: LSHConfig instance

    Returns:
        ConsistentHashRing with all table nodes added.
    """
    ring = ConsistentHashRing()

    for i in range(config.num_tables):
        node_id = f'table_{i}'
        ring.add_node(node_id, num_virtual_nodes=3)

    return ring


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Save Artifacts
# ─────────────────────────────────────────────────────────────────────────────

def save_lsh_artifacts(config, projection_matrices, ring, output_dir):
    """
    Save three files that all other team members will load:

        lsh_config.json          — hyperparameters (JSON, human-readable)
        projection_matrices.pkl  — list of 10 numpy arrays (binary pickle)
        consistent_hash_ring.pkl — ring state dict (binary pickle)

    Args:
        config:              LSHConfig instance
        projection_matrices: list of numpy arrays from generate_projection_matrices()
        ring:                ConsistentHashRing from build_ring()
        output_dir:          directory path string, e.g. 'lsh_index/'
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. lsh_config.json ───────────────────────────────────────────────
    config_path = os.path.join(output_dir, 'lsh_config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Saved: {config_path}")

    # ── 2. projection_matrices.pkl ───────────────────────────────────────
    matrices_path = os.path.join(output_dir, 'projection_matrices.pkl')
    with open(matrices_path, 'wb') as f:
        pickle.dump(projection_matrices, f, protocol=4)
    size_kb = os.path.getsize(matrices_path) / 1024
    print(f"  Saved: {matrices_path}  ({size_kb:.1f} KB)")

    # ── 3. consistent_hash_ring.pkl ──────────────────────────────────────
    ring_path = os.path.join(output_dir, 'consistent_hash_ring.pkl')
    with open(ring_path, 'wb') as f:
        pickle.dump(ring.to_dict(), f, protocol=4)
    size_kb = os.path.getsize(ring_path) / 1024
    print(f"  Saved: {ring_path}  ({size_kb:.1f} KB)")

    print("\nAll Task 1 artifacts saved successfully.")
    print("NEXT ACTION: Upload lsh_config.json to Google Drive now so teammates can start.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Run When Executed Directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Run this from the REPO ROOT:
    #     python lsh_index/lsh_structure.py

    print("=" * 60)
    print("Task 1: LSH Structure and Consistent Hashing")
    print("=" * 60)

    OUTPUT_DIR   = 'lsh_index'
    DATA_PATH    = 'data/all_cnn_embeddings.h5'

    # ── Step 1: Create config and show it ────────────────────────────────
    print("\n[1/5] Creating LSHConfig...")
    config = LSHConfig()
    print(f"  {config}")

    # ── Step 2: Generate projection matrices ─────────────────────────────
    print("\n[2/5] Generating projection matrices...")
    matrices = generate_projection_matrices(config)
    print(f"  Generated {len(matrices)} matrices")
    print(f"  Each matrix shape: {matrices[0].shape}")
    print(f"  First matrix sample values: {matrices[0][0, :5]}")
    print(f"  Last  matrix sample values: {matrices[-1][0, :5]}")
    assert len(matrices) == config.num_tables, "Wrong number of matrices!"
    assert matrices[0].shape == (config.hash_size, config.embedding_dim), "Wrong shape!"
    print("  Shape checks PASSED")

    # ── Step 3: Verify hash consistency ──────────────────────────────────
    print("\n[3/5] Verifying hash functions...")
    rng_test = np.random.default_rng(999)
    test_embedding = rng_test.standard_normal(128).astype(np.float32)

    for i, matrix in enumerate(matrices):
        verify_hash_consistency(test_embedding, matrix)

    print(f"  Single-vs-batch consistency verified for all {config.num_tables} tables")

    # Show sample hashes for the test embedding
    print("\n  Sample bucket keys for a random test embedding across all 10 tables:")
    for i, matrix in enumerate(matrices):
        key = hash_embedding(test_embedding, matrix)
        print(f"    table_{i}: {key}")

    # ── Step 4: Build and verify the consistent hash ring ────────────────
    print("\n[4/5] Building consistent hash ring...")
    ring = build_ring(config)
    print(f"  {ring}")

    # Verify distribution — each table should get roughly 1/10 of lookups
    test_keys = [hash_embedding(rng_test.standard_normal(128).astype(np.float32), matrices[0])
                 for _ in range(1000)]
    distribution = {}
    for key in test_keys:
        node = ring.get_node(key)
        distribution[node] = distribution.get(node, 0) + 1

    print(f"\n  Ring distribution over 1000 random bucket keys (expected ~100 each):")
    for node in sorted(distribution.keys()):
        bar = '█' * (distribution[node] // 5)
        print(f"    {node:10s}: {distribution[node]:4d}  {bar}")

    # ── Step 5: Run verification against real embeddings if available ─────
    print("\n[5/5] Running verification against real embeddings...")

    if not os.path.exists(DATA_PATH):
        print(f"  WARNING: {DATA_PATH} not found — skipping real-data verification.")
        print("  Run this check manually after downloading all_cnn_embeddings.h5")
    else:
        import h5py

        with h5py.File(DATA_PATH, 'r') as f:
            emb_shape = f['embeddings'].shape
            id_shape  = f['image_ids'].shape
            print(f"  embeddings shape: {emb_shape}")
            print(f"  image_ids  shape: {id_shape}")
            assert emb_shape == (1000000, 128), f"Expected (1000000, 128), got {emb_shape}"
            assert id_shape  == (1000000,),     f"Expected (1000000,), got {id_shape}"
            print("  Shape assertions PASSED")

            # Load 5 embeddings to test hashing
            sample_embeddings = f['embeddings'][:5]
            sample_ids        = f['image_ids'][:5]

        print(f"\n  Bucket keys for image IDs {sample_ids.tolist()} across all 10 tables:")
        for idx, (img_id, emb) in enumerate(zip(sample_ids, sample_embeddings)):
            keys = [hash_embedding(emb, m) for m in matrices]
            unique_keys = len(set(keys))
            print(f"    ID {img_id:6d}: {' | '.join(keys[:3])} ... ({unique_keys}/10 unique)")

        # Check that tables do NOT all produce the same key (seeding sanity check)
        emb0  = sample_embeddings[0]
        keys0 = [hash_embedding(emb0, m) for m in matrices]
        assert len(set(keys0)) > 1, (
            "All 10 tables produced IDENTICAL keys for the same embedding!\n"
            "This means the projection matrices are all the same — seeding bug!"
        )
        print("\n  Tables produce different keys for the same image — PASSED")

        # Compare two similar and two dissimilar embeddings
        emb_a = sample_embeddings[0]
        emb_b = sample_embeddings[1]
        emb_z = sample_embeddings[4]

        keys_a = set(hash_embedding(emb_a, m) for m in matrices)
        keys_b = set(hash_embedding(emb_b, m) for m in matrices)
        keys_z = set(hash_embedding(emb_z, m) for m in matrices)

        shared_ab = sum(1 for m in matrices
                        if hash_embedding(emb_a, m) == hash_embedding(emb_b, m))
        shared_az = sum(1 for m in matrices
                        if hash_embedding(emb_a, m) == hash_embedding(emb_z, m))

        cos_ab = float(np.dot(emb_a, emb_b) /
                       (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
        cos_az = float(np.dot(emb_a, emb_z) /
                       (np.linalg.norm(emb_a) * np.linalg.norm(emb_z)))

        print(f"\n  Pair (ID 0, ID 1): cosine similarity={cos_ab:.4f}, "
              f"tables sharing a bucket: {shared_ab}/10")
        print(f"  Pair (ID 0, ID 4): cosine similarity={cos_az:.4f}, "
              f"tables sharing a bucket: {shared_az}/10")

    # ── Save everything ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Saving Task 1 artifacts...")
    print('=' * 60)
    save_lsh_artifacts(config, matrices, ring, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("Task 1 COMPLETE. Files produced:")
    print(f"  lsh_index/lsh_config.json")
    print(f"  lsh_index/projection_matrices.pkl")
    print(f"  lsh_index/consistent_hash_ring.pkl")
    print(f"\nIMMEDIATE ACTION: Upload lsh_index/lsh_config.json to")
    print(f"PDC_Project_Shared/lsh_index/ on Google Drive.")
    print('=' * 60)