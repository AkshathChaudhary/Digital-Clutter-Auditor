"""Unit test for high-performance video deduplication (dedup/video.py).

Verifies:
1. Video keyframe sampling and 64-bit integer perceptual hashing.
2. Fast bitwise Hamming distance and O(1) duration filtering.
3. Near-duplicate matching between original and recompressed/scaled videos.
4. Correct rejection of unrelated videos and duration mismatches.
5. End-to-end SQLite clustering into `files.near_group`.
"""

import gc
import sys
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import db
from dedup.video import (
    compare_videos,
    compute_video_phash,
    find_video_duplicates,
)
from models import FileRecord


def create_synthetic_video(
    file_path: Path,
    num_frames: int = 30,
    fps: int = 10,
    scene_type: str = "circle",
    resolution: tuple = (128, 128),
) -> None:
    """Generate a valid synthetic video file with proportional coordinates across resolutions."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, resolution)
    w, h = resolution

    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        if scene_type == "circle":
            # Normalized proportional motion: circle moves horizontally across the screen
            norm_x = 0.2 + (i / num_frames) * 0.6
            cx = int(norm_x * w)
            cy = h // 2
            radius = int(0.14 * min(w, h))
            cv2.circle(frame, (cx, cy), radius, (220, 80, 20), -1)

        elif scene_type == "circle_noisy":
            # Identical proportional motion with slight color variance and noise (recompression artifact)
            norm_x = 0.2 + (i / num_frames) * 0.6
            cx = int(norm_x * w)
            cy = h // 2
            radius = int(0.14 * min(w, h))
            cv2.circle(frame, (cx, cy), radius, (215, 82, 25), -1)
            noise = np.random.randint(0, 8, frame.shape, dtype=np.uint8)
            frame = cv2.add(frame, noise)

        elif scene_type == "stripes":
            # Completely different visual content: horizontal stripes
            for y in range(0, h, 16):
                frame[y : y + 8, :] = (240, 240, 240)

        out.write(frame)

    out.release()


def test_video_phash_and_matching():
    print("Testing video keyframe hashing and bitwise matching...")

    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmpdir:
        tmp_path = Path(tmpdir)
        vid_original = tmp_path / "original.mp4"
        vid_recompressed = tmp_path / "recompressed_scaled.mp4"
        vid_different = tmp_path / "unrelated_scene.mp4"
        vid_short = tmp_path / "short_clip.mp4"

        # 1. Create synthetic test videos:
        # vid_original: 30 frames (3s) at 128x128
        create_synthetic_video(vid_original, num_frames=30, fps=10, scene_type="circle", resolution=(128, 128))
        # vid_recompressed: Same scene but smaller resolution (80x80) with noise (simulating mobile share/compression)
        create_synthetic_video(vid_recompressed, num_frames=30, fps=10, scene_type="circle_noisy", resolution=(80, 80))
        # vid_different: Completely different visual content (stripes)
        create_synthetic_video(vid_different, num_frames=30, fps=10, scene_type="stripes", resolution=(128, 128))
        # vid_short: Only 10 frames (1s) of moving circle (duration mismatch)
        create_synthetic_video(vid_short, num_frames=10, fps=10, scene_type="circle", resolution=(128, 128))

        # 2. Extract signatures
        dur1, h1 = compute_video_phash(vid_original, num_samples=5)
        dur2, h2 = compute_video_phash(vid_recompressed, num_samples=5)
        dur3, h3 = compute_video_phash(vid_different, num_samples=5)
        dur_short, h_short = compute_video_phash(vid_short, num_samples=5)

        assert dur1 is not None and len(h1) >= 3, "Failed to extract original video signature!"
        assert dur2 is not None and len(h2) >= 3, "Failed to extract recompressed video signature!"
        assert dur3 is not None and len(h3) >= 3, "Failed to extract different video signature!"

        # Compute and print keyframe distances for transparency
        dists = [(a ^ b).bit_count() for a, b in zip(h1, h2)]
        avg_d = sum(dists) / len(dists)
        print(f"  Keyframe Bitwise Distances (Original vs Recompressed): {dists} (Average: {avg_d:.2f})")

        diff_dists = [(a ^ b).bit_count() for a, b in zip(h1, h3)]
        avg_diff = sum(diff_dists) / len(diff_dists)
        print(f"  Keyframe Bitwise Distances (Original vs Stripes): {diff_dists} (Average: {avg_diff:.2f})")

        # 3. Test Near-Duplicate Match: Original vs Recompressed/Scaled
        is_dup = compare_videos(h1, h2, duration_a=dur1, duration_b=dur2, max_avg_dist=10.0)
        assert is_dup is True, f"Original and recompressed videos should match! (avg distance was {avg_d})"

        # 4. Test Unrelated Scene: Original vs Stripes
        is_diff = compare_videos(h1, h3, duration_a=dur1, duration_b=dur3, max_avg_dist=10.0)
        assert is_diff is False, "Unrelated video with different visual scene should NOT match!"

        # 5. Test O(1) Duration Gating: Original (3s) vs Short Clip (1s)
        is_duration_mismatch = compare_videos(h1, h_short, duration_a=dur1, duration_b=dur_short, max_avg_dist=10.0)
        assert is_duration_mismatch is False, "Videos with severe duration mismatch must be rejected by duration filter!"

    print(" Video hashing & comparison unit tests passed!")


def test_find_video_duplicates_end_to_end():
    print("\nTesting find_video_duplicates() database integration...")

    temp_dir_kwargs = {}
    if sys.version_info >= (3, 10):
        temp_dir_kwargs["ignore_cleanup_errors"] = True

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmpdir:
        tmp_path = Path(tmpdir)
        test_db = tmp_path / "test_clutteriq.db"
        db.init_db(test_db)

        # Create video files
        v1_path = tmp_path / "holiday_clip.mp4"
        v2_path = tmp_path / "holiday_clip_compressed.mp4"
        v3_path = tmp_path / "random_tutorial.mp4"

        create_synthetic_video(v1_path, num_frames=30, scene_type="circle", resolution=(128, 128))
        create_synthetic_video(v2_path, num_frames=30, scene_type="circle_noisy", resolution=(80, 80))
        create_synthetic_video(v3_path, num_frames=30, scene_type="stripes", resolution=(128, 128))

        now = time.time()
        r1 = FileRecord(path=str(v1_path), name="holiday_clip.mp4", ext="mp4", kind="video", size=50000, modified=now)
        r2 = FileRecord(path=str(v2_path), name="holiday_clip_compressed.mp4", ext="mp4", kind="video", size=35000, modified=now)
        r3 = FileRecord(path=str(v3_path), name="random_tutorial.mp4", ext="mp4", kind="video", size=60000, modified=now)

        id1 = db.upsert_file(r1, db_path=test_db)
        id2 = db.upsert_file(r2, db_path=test_db)
        id3 = db.upsert_file(r3, db_path=test_db)

        # Run video duplicate clustering
        num_groups = find_video_duplicates(db_path=test_db, num_samples=5, max_avg_dist=10.0, show_progress=False)
        assert num_groups == 1, f"Expected 1 video duplicate group, got {num_groups}"

        f1 = db.get_file_by_id(id1, db_path=test_db)
        f2 = db.get_file_by_id(id2, db_path=test_db)
        f3 = db.get_file_by_id(id3, db_path=test_db)

        assert f1.near_group is not None, "Video 1 should have near_group assigned!"
        assert f2.near_group is not None, "Video 2 should have near_group assigned!"
        assert f1.near_group == f2.near_group, "Video 1 and Video 2 must share the same near_group ID!"
        assert f3.near_group is None, "Video 3 (unrelated) should NOT be in any group!"

        del f1, f2, f3, r1, r2, r3
        gc.collect()

    print(" Database integration tests passed successfully!")


if __name__ == "__main__":
    test_video_phash_and_matching()
    test_find_video_duplicates_end_to_end()
