import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ssl_fix
from scanner.walker import scan_folder
from dedup.exact import find_exact_duplicates
from dedup.perceptual import find_near_duplicates
from quality.blur_rank import choose_keepers
from dedup.semantic import compute_embeddings, cluster_semantic
from privacy.sensitive import scan_sensitive
from ml.junk_model import predict_scores
from config import DB_PATH
import db

demo_folder = Path('demo_showcase')
assert demo_folder.exists(), 'demo_showcase folder does not exist!'

print(f'1. Scanning folder {demo_folder}...')
scan_res = scan_folder(str(demo_folder))
print(f'   Scanned {scan_res} files.')

print('2. Running Exact Deduplication...')
exact_res = find_exact_duplicates(db_path=DB_PATH)
print(f'   Exact duplicates found: {len(exact_res)}')

print('3. Running Perceptual Deduplication (pHash)...')
near_res = find_near_duplicates(db_path=DB_PATH)
print(f'   Near duplicate groups found: {len(near_res)}')

print('4. Running Quality Ranking & Keeper Selection...')
keepers = choose_keepers(db_path=DB_PATH)
print(f'   Keepers selected: {len(keepers)}')

print('5. Computing CLIP Semantic Embeddings...')
compute_embeddings(db_path=DB_PATH, show_progress=False)

print('6. Clustering Semantic Groups (HNSW)...')
sem_groups = cluster_semantic(db_path=DB_PATH)
print(f'   Semantic clusters found: {sem_groups}')

print('7. Scanning Sensitive Documents...')
sens_res = scan_sensitive(db_path=DB_PATH, show_progress=False)
print(f'   Sensitive items detected: {sens_res}')

print('8. Predicting ML Junk Scores...')
ml_res = predict_scores(db_path=DB_PATH)
print(f'   ML scored items: {ml_res}')

print('9. Verifying KPI queries...')
total_files, total_bytes, recoverable_bytes, total_groups, privacy_alerts = db.get_overview_kpis(db_path=DB_PATH)
print(f'   Overview KPIs: Files={total_files}, Storage={total_bytes}, Recoverable={recoverable_bytes}, Groups={total_groups}, Alerts={privacy_alerts}')
assert total_files >= 20, f"Expected at least 20 files, got {total_files}"
assert privacy_alerts >= 1, "Expected at least 1 privacy alert"
assert total_groups >= 3, "Expected duplicate groups"

print('\nEND-TO-END PIPELINE DIAGNOSTIC: 100% OPERATIONAL & SUCCESSFUL!')
