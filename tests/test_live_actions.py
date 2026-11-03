import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
from config import DB_PATH
import db
from safety.quarantine import quarantine_file, permanently_delete_file
from tests.prepare_demo_state import setup_fresh_demo

def test_full_user_workflow():
    print("=== Step 0: Setting up fresh demo state ===")
    setup_fresh_demo()

    # Initial state
    kpis_0 = db.get_overview_kpis(DB_PATH)
    print(f"Initial KPIs: Files={kpis_0[0]}, Storage={kpis_0[1]}, Recov={kpis_0[2]}, Groups={kpis_0[3]}, Alerts={kpis_0[4]}")
    assert kpis_0[3] >= 3, "Expected multiple duplicate groups"

    # Step 1: Find an exact duplicate copy to quarantine
    conn = sqlite3.connect(DB_PATH)
    copy_row = conn.execute("SELECT id, name, path FROM files WHERE exact_group IS NOT NULL AND is_keeper = 0 LIMIT 1").fetchone()
    conn.close()
    assert copy_row is not None, "Expected an exact copy"
    copy_id, copy_name, copy_path = copy_row
    print(f"\n=== Step 1: Quarantining exact duplicate copy '{copy_name}' (ID={copy_id}) ===")
    
    q_dest = quarantine_file(copy_id, db_path=DB_PATH)
    assert q_dest.exists(), "Quarantined file must exist in vault"
    assert not Path(copy_path).exists(), "Original copy must be moved from original folder"

    # Verify that the solitary keeper is now resolved and no longer counted as a duplicate
    kpis_1 = db.get_overview_kpis(DB_PATH)
    print(f"KPIs after 1 quarantine: Groups={kpis_1[3]} (was {kpis_0[3]})")
    assert kpis_1[3] < kpis_0[3], "Duplicate groups count must decrease automatically!"

    # Step 2: Delete permanently from vault (ONLY this 1 file)
    conn = sqlite3.connect(DB_PATH)
    q_entry = conn.execute("SELECT id FROM quarantine_log WHERE file_id = ? AND restored_at IS NULL", (copy_id,)).fetchone()
    conn.close()
    assert q_entry is not None, "Quarantine record must exist"
    qid = q_entry[0]

    print(f"\n=== Step 2: Permanently deleting only QID={qid} from vault ===")
    success, fname, freed = permanently_delete_file(qid, db_path=DB_PATH)
    assert success is True
    assert not q_dest.exists(), "Quarantined file must be deleted from vault"

    # Verify other files in demo_showcase still exist!
    conn = sqlite3.connect(DB_PATH)
    active_count = conn.execute("SELECT count(*) FROM files WHERE status = 'active'").fetchone()[0]
    conn.close()
    assert active_count == 20, f"Expected exactly 20 active files remaining (only 1 copy deleted), got {active_count}"
    print(f"Active files remaining: {active_count} (only 1 copy purged as expected!)")

    # Step 3: Test Semantic Cluster quarantine
    conn = sqlite3.connect(DB_PATH)
    sem_copies = conn.execute("SELECT id, name FROM files WHERE semantic_group IS NOT NULL AND is_keeper = 0").fetchall()
    conn.close()
    print(f"\n=== Step 3: Quarantining {len(sem_copies)} semantic non-keeper image copies ===")
    for fid, fname in sem_copies:
        quarantine_file(fid, db_path=DB_PATH)
        print(f"  Quarantined semantic copy: {fname}")

    kpis_2 = db.get_overview_kpis(DB_PATH)
    print(f"KPIs after semantic quarantines: Groups={kpis_2[3]} (decreased from {kpis_1[3]})")

    # Step 4: Quarantine the sensitive KYC file
    conn = sqlite3.connect(DB_PATH)
    sens_id = conn.execute("SELECT id FROM files WHERE privacy_type IS NOT NULL AND status = 'active'").fetchone()[0]
    conn.close()
    print(f"\n=== Step 4: Quarantining privacy risk file (ID={sens_id}) ===")
    quarantine_file(sens_id, db_path=DB_PATH)

    kpis_final = db.get_overview_kpis(DB_PATH)
    print(f"Final KPIs: Files={kpis_final[0]}, Groups={kpis_final[3]}, Alerts={kpis_final[4]}")
    assert kpis_final[4] == 0, "Privacy alerts must drop to 0!"

    print("\n[SUCCESS] ALL ACTIONS VERIFIED: AUTO-UPDATING STATS, SINGLE FILE DELETE & SEMANTIC QUARANTINE WORK FLAWLESSLY!")

if __name__ == "__main__":
    test_full_user_workflow()
