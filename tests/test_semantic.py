"""Unit test for CLIP semantic deduplication (dedup/semantic.py).

Verifies:
1. Disjoint Set Union (Union-Find) transitivity and path compression.
2. CLIP embedding computation, L2 normalization, and disk caching.
3. Cosine similarity clustering into semantic duplicate groups.
4. Zero-shot category tagging.
"""

# Must be line 1 to apply descriptor patch before any networking imports
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ssl_fix

import gc
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

from config import CLIP_SIM_THRESHOLD
import db
from dedup.semantic import (
    UnionFind,
    cluster_semantic,
    compute_embeddings,
    tag_categories,
)
from models import FileRecord


def test_union_find():
    """Verify Union-Find clusters transitively with path compression."""
    print("Testing Union-Find data structure...")
    uf = UnionFind([1, 2, 3, 4, 5])
    uf.union(1, 2)
    uf.union(2, 3)  # 1, 2, 3 are now in the same component
    uf.union(4, 5)

    assert uf.find(1) == uf.find(3), "Transitive connection 1 ~ 3 failed!"
    assert uf.find(1) != uf.find(4), "Independent sets 1 and 4 should not be connected!"

    comps = uf.get_components()
    group_sizes = sorted([len(v) for v in comps.values()])
    assert group_sizes == [2, 3], f"Expected group sizes [2, 3], got {group_sizes}"
    print(" Union-Find tests passed!")


def test_semantic_pipeline():
    """Verify embedding computation, caching, clustering, and categorization."""
    print("\nTesting CLIP Semantic Pipeline on synthetic images...")

    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmpdir:
        tmp_path = Path(tmpdir)
        test_db = tmp_path / "test_clutteriq.db"
        test_emb = tmp_path / "embeddings.npy"

        # 1. Initialize DB
        db.init_db(test_db)

        # 2. Create synthetic test images with distinct visual semantics:
        # img1 and img2: Blue circle on white background (visually & semantically near-duplicates)
        # img3: High-contrast striped pattern (semantically completely different)
        img1_path = tmp_path / "blue_circle_1.png"
        img2_path = tmp_path / "blue_circle_2.png"
        img3_path = tmp_path / "striped_pattern.png"

        # Image 1: White canvas with blue circle
        im1 = Image.new("RGB", (128, 128), color=(255, 255, 255))
        d1 = ImageDraw.Draw(im1)
        d1.ellipse([25, 25, 103, 103], fill=(20, 60, 220))
        im1.save(str(img1_path))

        # Image 2: Near-duplicate (blue circle with 1-pixel shift and slight color tone difference)
        im2 = Image.new("RGB", (128, 128), color=(255, 255, 255))
        d2 = ImageDraw.Draw(im2)
        d2.ellipse([26, 26, 104, 104], fill=(22, 62, 218))
        im2.save(str(img2_path))

        # Image 3: High-contrast black and white horizontal stripes (completely different scene)
        im3 = Image.new("RGB", (128, 128), color=(0, 0, 0))
        d3 = ImageDraw.Draw(im3)
        for y in range(0, 128, 16):
            d3.rectangle([0, y, 128, y + 8], fill=(255, 255, 255))
        im3.save(str(img3_path))

        now = time.time()
        r1 = FileRecord(path=str(img1_path), name="circle1.png", ext="png", kind="image", size=2000, modified=now)
        r2 = FileRecord(path=str(img2_path), name="circle2.png", ext="png", kind="image", size=2000, modified=now)
        r3 = FileRecord(path=str(img3_path), name="stripes.png", ext="png", kind="image", size=2000, modified=now)

        id1 = db.upsert_file(r1, db_path=test_db)
        id2 = db.upsert_file(r2, db_path=test_db)
        id3 = db.upsert_file(r3, db_path=test_db)

        # 3. Compute Embeddings
        print(" Computing embeddings with CLIP...")
        new_count = compute_embeddings(db_path=test_db, emb_path=test_emb, show_progress=False)
        assert new_count == 3, f"Expected 3 images embedded, got {new_count}"
        assert test_emb.exists(), "embeddings.npy was not created!"

        matrix = np.load(str(test_emb))
        assert matrix.shape == (3, 512), f"Expected shape (3, 512), got {matrix.shape}"

        # Verify unit L2 normalization (length of each vector should be ~1.0)
        norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"Vectors are not L2 normalized: {norms}"

        # Display pairwise cosine similarity
        sim_matrix = np.dot(matrix, matrix.T)
        print(f" Pairwise Cosine Similarities:")
        print(f"   Circle 1 vs Circle 2:  {sim_matrix[0, 1]:.4f} (Expected high > 0.90)")
        print(f"   Circle 1 vs Stripes:   {sim_matrix[0, 2]:.4f} (Expected low < 0.85)")

        # 4. Verify Caching (re-run should embed 0 new images)
        print(" Verifying embedding cache...")
        re_run_count = compute_embeddings(db_path=test_db, emb_path=test_emb, show_progress=False)
        assert re_run_count == 0, f"Expected 0 re-computations due to cache, got {re_run_count}"

        # 5. Cluster Semantics using project threshold
        print(f" Clustering semantic duplicates (threshold={CLIP_SIM_THRESHOLD})...")
        num_groups = cluster_semantic(db_path=test_db, emb_path=test_emb, sim_threshold=CLIP_SIM_THRESHOLD)
        assert num_groups == 1, f"Expected 1 semantic group, got {num_groups}"

        f1 = db.get_file_by_id(id1, db_path=test_db)
        f2 = db.get_file_by_id(id2, db_path=test_db)
        f3 = db.get_file_by_id(id3, db_path=test_db)

        assert f1.semantic_group is not None, "Circle 1 should be in a group"
        assert f2.semantic_group is not None, "Circle 2 should be in a group"
        assert f1.semantic_group == f2.semantic_group, "Circle 1 and Circle 2 should share the same group ID!"
        assert f3.semantic_group is None, "Striped pattern should NOT be in any group!"

        # 6. Test Zero-Shot Categorization
        print(" Testing zero-shot categorization...")
        tagged = tag_categories(db_path=test_db, emb_path=test_emb)
        assert tagged == 3, f"Expected 3 images tagged, got {tagged}"
        f1_tagged = db.get_file_by_id(id1, db_path=test_db)
        assert f1_tagged.category is not None, "Category should not be empty!"
        print(f"  Tagged category for Circle 1: '{f1_tagged.category}'")

        # Clean references for Windows
        del matrix, sim_matrix, f1, f2, f3, f1_tagged, r1, r2, r3, im1, im2, im3, d1, d2, d3
        gc.collect()

    print("\n All CLIP Semantic Deduplication tests passed successfully!")


if __name__ == "__main__":
    test_union_find()
    test_semantic_pipeline()
