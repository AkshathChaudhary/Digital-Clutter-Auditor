import db
from config import DB_PATH

# Initialize the database if necessary.
db.init_db(DB_PATH)

# Retrieve all active records.
records = db.get_active_files(db_path=DB_PATH)

print("\n========== DATABASE CONTENTS ==========")

for record in records:
    print(
        f"ID: {record.id} | "
        f"Name: {record.name} | "
        f"Type: {record.kind} | "
        f"Size: {record.size} bytes"
    )

print("\nTotal records:", len(records))