import sys
from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ssl_fix
from config import DB_PATH
import db
from scanner.walker import scan_folder
from dedup.exact import find_exact_duplicates
from dedup.perceptual import find_near_duplicates
from quality.blur_rank import choose_keepers
from dedup.semantic import compute_embeddings, cluster_semantic
from privacy.sensitive import scan_sensitive
from ml.junk_model import predict_scores

def setup_fresh_demo():
    print("Preparing clean-slate demo audit on demo_showcase...")
    # Initialize DB schema
    db.init_db(DB_PATH)
    
    # Clean old rows from files and reset
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM files")
    conn.execute("DELETE FROM quarantine_log")
    conn.commit()
    conn.close()

    demo_dir = PROJECT_ROOT / "demo_showcase"
    assert demo_dir.exists(), f"Directory not found: {demo_dir}"

    print(f"1. Scanning {demo_dir}...")
    num_scanned = scan_folder(str(demo_dir))
    print(f"   -> Scanned {num_scanned} files.")

    print("2. Detecting Exact SHA-256 Duplicates...")
    exact_res = find_exact_duplicates(db_path=DB_PATH)
    print(f"   -> Found exact duplicate groups.")

    print("3. Detecting Perceptual (pHash) Duplicates...")
    near_res = find_near_duplicates(db_path=DB_PATH)
    print(f"   -> Found near duplicate groups.")

    print("4. Selecting Best Quality Keepers (Blur Variance)...")
    keepers = choose_keepers(db_path=DB_PATH)
    print(f"   -> Designated {len(keepers)} keepers.")

    print("5. Generating CLIP Semantic Embeddings...")
    compute_embeddings(db_path=DB_PATH, show_progress=False)

    print("6. Clustering Semantic Images (HNSW Cosine Graph)...")
    sem_count = cluster_semantic(db_path=DB_PATH)
    print(f"   -> Found {sem_count} semantic clusters.")

    print("7. Scanning Privacy Risks (Aadhaar & Cards)...")
    privacy_res = scan_sensitive(db_path=DB_PATH, show_progress=False)
    print(f"   -> Detected {privacy_res} privacy findings.")

    print("8. Scoring ML Junk Probabilities...")
    scored = predict_scores(db_path=DB_PATH)
    print(f"   -> Scored {scored} files.")

    # Telemetry verification
    conn = sqlite3.connect(DB_PATH)
    total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    exact_groups = conn.execute("SELECT COUNT(DISTINCT exact_group) FROM files WHERE exact_group IS NOT NULL").fetchone()[0]
    near_groups = conn.execute("SELECT COUNT(DISTINCT near_group) FROM files WHERE near_group IS NOT NULL").fetchone()[0]
    sem_groups = conn.execute("SELECT COUNT(DISTINCT semantic_group) FROM files WHERE semantic_group IS NOT NULL").fetchone()[0]
    privacy_alerts = conn.execute("SELECT COUNT(*) FROM files WHERE privacy_type IS NOT NULL").fetchone()[0]
    conn.close()

    print("\n==========================================")
    print("DEMO AUDIT READY FOR PRESENTATION:")
    print(f"  Total Files Audited: {total_files}")
    print(f"  Exact Duplicate Clusters: {exact_groups}")
    print(f"  Perceptual Near Clusters: {near_groups}")
    print(f"  Semantic CLIP Clusters: {sem_groups}")
    print(f"  Critical Privacy Alerts: {privacy_alerts}")
    print("==========================================\n")

if __name__ == "__main__":
    setup_fresh_demo()

