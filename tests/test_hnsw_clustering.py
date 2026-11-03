"""Quick test to verify cluster_semantic with HNSW graph search and fallback paths.

Verifies:
1. Synthesizes 120 vector embeddings (N >= 100 to trigger the HNSW threshold).
2. Tests that duplicate pairs (similarity > 0.95) are clustered correctly.
3. Tests that random noise vectors (unrelated) are NOT clustered.
4. Verifies there are no syntax errors or typos (e.g. np.arange).
5. Ensures graceful operation whether hnswlib is installed or not.
"""

import gc
import sys
from pathlib import Path
import tempfile
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ssl_fix

import db
from dedup.semantic import cluster_semantic
from models import FileRecord


def test_hnsw_clustering_large_scale():
    print("Testing cluster_semantic() on N >= 100 vectors...")

    # Check if hnswlib is available in the environment
    try:
        import hnswlib
        print("  Detected hnswlib installed -> Testing HNSW O(log N) Graph Path!")
    except ImportError:
        print("  hnswlib not installed -> Testing graceful NumPy fallback path!")

    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmpdir:
        tmp_path = Path(tmpdir)
        test_db = tmp_path / "test_clutteriq.db"
        test_emb = tmp_path / "embeddings.npy"

        # 1. Initialize SQLite database
        db.init_db(test_db)

        # 2. Synthesize 120 normalized 512-dim vectors (N >= 100)
        rng = np.random.RandomState(42)
        dim = 512
        N = 120

        # Create random base unit vectors
        raw_vectors = rng.randn(N, dim).astype(np.float32)
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        vectors = raw_vectors / norms

        # Create near-duplicate pairs (0 & 1) and (10 & 11) with guaranteed similarity ~0.96
        # Blend original vector with orthogonal noise:
        noise1 = rng.randn(dim).astype(np.float32)
        noise1 -= np.dot(noise1, vectors[0]) * vectors[0]  # Make strictly orthogonal
        noise1 /= np.linalg.norm(noise1)
        vectors[1] = 0.96 * vectors[0] + np.sqrt(1 - 0.96**2) * noise1

        noise2 = rng.randn(dim).astype(np.float32)
        noise2 -= np.dot(noise2, vectors[10]) * vectors[10]
        noise2 /= np.linalg.norm(noise2)
        vectors[11] = 0.96 * vectors[10] + np.sqrt(1 - 0.96**2) * noise2

        # Verify synthetic pair similarities
        sim_0_1 = float(np.dot(vectors[0], vectors[1]))
        sim_10_11 = float(np.dot(vectors[10], vectors[11]))
        print(f"  Synthetic Pair 0 & 1 Cosine Similarity:   {sim_0_1:.4f} (Expected ~0.96)")
        print(f"  Synthetic Pair 10 & 11 Cosine Similarity: {sim_10_11:.4f} (Expected ~0.96)")
        assert sim_0_1 >= 0.92, f"Pair 0 & 1 similarity was {sim_0_1}, expected >= 0.92"
        assert sim_10_11 >= 0.92, f"Pair 10 & 11 similarity was {sim_10_11}, expected >= 0.92"

        # Save synthetic vectors to disk
        np.save(str(test_emb), vectors)

        # 3. Insert 120 dummy file records into SQLite matching the embedding indices
        now = time.time()
        file_ids = []
        for i in range(N):
            rec = FileRecord(
                path=str(tmp_path / f"photo_{i:03d}.jpg"),
                name=f"photo_{i:03d}.jpg",
                ext="jpg",
                kind="image",
                size=2048,
                modified=now,
                emb_index=i,  # Matches row index in embeddings.npy
            )
            fid = db.upsert_file(rec, db_path=test_db)
            file_ids.append(fid)

        # 4. Run cluster_semantic on 120 vectors
        print("  Running cluster_semantic()...")
        num_groups = cluster_semantic(db_path=test_db, emb_path=test_emb, sim_threshold=0.92)

        print(f"  Successfully formed {num_groups} duplicate groups!")
        assert num_groups >= 2, f"Expected at least 2 duplicate groups, got {num_groups}"

        # 5. Verify that pair (0, 1) and pair (10, 11) are properly grouped
        rec0 = db.get_file_by_id(file_ids[0], db_path=test_db)
        rec1 = db.get_file_by_id(file_ids[1], db_path=test_db)
        rec10 = db.get_file_by_id(file_ids[10], db_path=test_db)
        rec11 = db.get_file_by_id(file_ids[11], db_path=test_db)

        # Unrelated random vector (e.g. index 50)
        rec50 = db.get_file_by_id(file_ids[50], db_path=test_db)

        assert rec0.semantic_group is not None, "Photo 0 should be in a group"
        assert rec1.semantic_group is not None, "Photo 1 should be in a group"
        assert rec0.semantic_group == rec1.semantic_group, "Photo 0 and 1 must share the same semantic_group ID!"

        assert rec10.semantic_group is not None, "Photo 10 should be in a group"
        assert rec11.semantic_group is not None, "Photo 11 should be in a group"
        assert rec10.semantic_group == rec11.semantic_group, "Photo 10 and 11 must share the same semantic_group ID!"

        assert rec0.semantic_group != rec10.semantic_group, "Unrelated pairs must have different group IDs!"
        assert rec50.semantic_group is None, "Random photo 50 should NOT be grouped with anything!"

        del rec0, rec1, rec10, rec11, rec50, vectors, raw_vectors
        gc.collect()

    print("\n Quick verification test PASSED with 0 errors!")


if __name__ == "__main__":
    test_hnsw_clustering_large_scale()
