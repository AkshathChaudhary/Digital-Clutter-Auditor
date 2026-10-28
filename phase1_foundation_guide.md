# Phase 1: Building the Foundation — `config.py` → `models.py` → `db.py`

These 3 files are the **backbone** of the entire project. Every other file imports from them. Get these right first.

---

## File 1: `config.py` — The Single Source of Truth

### Why this file?
Every real project has **magic numbers** scattered everywhere — thresholds, file paths, extension lists. If you hardcode them in 10 different files, changing one means hunting through all 10. `config.py` puts them **all in one place**.

> [!TIP]
> **Real-world lesson:** In production codebases, this is often a `.env` file or a YAML config. For this project, a simple Python file with constants is perfect.

### What it needs to contain

Think about what the rest of the system will need to know:

1. **Paths** — Where does the database live? Where do embeddings get saved? Where does the trained model go? Where is the quarantine folder?
2. **File type sets** — Which extensions are images? Which are documents? (The scanner needs this to classify files.)
3. **Thresholds** — How close do two pHashes need to be to count as "near-duplicate"? What cosine similarity means "semantically similar"? What blur score means "blurry"? How old is "stale"?
4. **Batch sizes** — How many images to embed at once with CLIP?

### Your blueprint

```python
# config.py — All project constants in one place

from pathlib import Path

# === File paths ===
# Where does the SQLite database live?
# Where do CLIP embeddings get saved (.npy file)?
# Where does the trained junk model get saved (.joblib file)?
# Where is the quarantine folder?

# === File type classification ===
# A set of image extensions (jpg, jpeg, png, webp, bmp, gif, tiff)
# A set of document extensions (pdf, doc, docx, txt, xlsx, pptx, csv)

# === Duplicate detection thresholds ===
# Max Hamming distance for pHash near-duplicates (start with 6)
# Min cosine similarity for CLIP semantic grouping (start with 0.92)

# === Quality thresholds ===
# Blur score below which an image is "blurry" (start with 100.0)
# Days after which a file is "stale" (start with 365)

# === Processing ===
# Minimum file size worth hashing (1 byte — skip 0-byte files)
# CLIP batch size (32 images at a time)
```

### Key decisions to think about
- **Why `pathlib.Path` instead of plain strings?** Try `Path("data") / "clutteriq.db"` in a Python shell. It handles `/` vs `\` for you across OS.
- **Why sets for extensions, not lists?** Lookup in a set is O(1), lookup in a list is O(n). When you're checking thousands of files, this matters.
- **Why are these just module-level variables and not a class?** Simplicity. A class would add ceremony with no benefit here. Constants don't need methods.

### How to test it
Open a Python shell and try:
```python
from config import *
print(DB_PATH)
print("jpg" in IMAGE_EXTS)  # Should be True
print("pdf" in IMAGE_EXTS)  # Should be False
```

### ✅ When you're done, show me your `config.py` and I'll review it.

---

## File 2: `models.py` — Your Data Shape

### Why this file?
When you pass file information between functions, you could use dictionaries — but then you'd have no idea what keys exist, no autocomplete, and typos like `file["naem"]` would silently return `None`. A **dataclass** gives you a structured object with named fields.

### Concept: Python Dataclasses

A dataclass auto-generates `__init__`, `__repr__`, and more:

```python
from dataclasses import dataclass

# Without dataclass — you write boilerplate:
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# With dataclass — Python writes it for you:
@dataclass
class Point:
    x: float
    y: float
```

Now `Point(3.0, 4.0)` just works, and `print(point)` shows `Point(x=3.0, y=4.0)`.

### What it needs to contain

A `FileRecord` dataclass that **mirrors the database schema**. Look at the `files` table in the spec (Section 5). Each column becomes a field.

### Your blueprint

```python
# models.py — Data structures shared across the project

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FileRecord:
    # Think about which fields are REQUIRED (every file has them)
    # vs OPTIONAL (filled in later by different pipeline stages)
    
    # Required fields (scanner fills these):
    # path, name, ext, kind, size, created, modified, accessed
    
    # Optional fields (filled by later stages):
    # id, sha256, phash, width, height, blur_score, emb_index
    # exact_group, near_group, semantic_group, is_keeper
    # category, junk_score, junk_reason
    # privacy_type, privacy_masked
    # user_label, status
    pass
```

### Key decisions to think about
- **`Optional[str]` vs `str`** — Most fields start as `None` (e.g., `sha256` is only computed for duplicate candidates). Use `Optional[X] = None` for these.
- **Default values** — `status` should default to `"active"`, `is_keeper` to `0`. In dataclasses, fields with defaults must come AFTER fields without defaults.
- **`id: Optional[int] = None`** — The database auto-generates the ID, so your code won't know it until after insertion.

### Hints
- You'll have roughly **22 fields** matching the 22 columns in the `files` table
- Use `float` for timestamps (epoch seconds), `int` for sizes, `str` for paths
- The order matters: required fields first, optional (defaulted) fields after

### How to test it
```python
from models import FileRecord

# Create a minimal record (only required fields)
f = FileRecord(
    path="C:/Users/test/photo.jpg",
    name="photo.jpg",
    ext="jpg",
    kind="image",
    size=1024000,
    created=1700000000.0,
    modified=1700000000.0,
    accessed=1700000000.0,
)
print(f)           # Should show all fields
print(f.sha256)    # Should be None
print(f.status)    # Should be "active"
```

### ✅ When you're done, show me your `models.py` and I'll review it.

---

## File 3: `db.py` — The Heart of the System

### Why this file?
Every module in the pipeline (scanner, dedup, quality, ML, privacy, UI) reads from and writes to **one shared SQLite database**. `db.py` is the gatekeeper — it creates the tables and provides helper functions so that no other file needs to write raw SQL strings.

> [!IMPORTANT]
> This is the most important file in Phase 1. If the schema is wrong, everything breaks. If the helpers are clunky, every other file becomes painful to write.

### Concepts to learn first

**SQLite** — a database engine that lives in a single file (no server to install). Python's `sqlite3` module is built-in.

Try this in a Python shell to get a feel:
```python
import sqlite3

conn = sqlite3.connect("test.db")     # Creates/opens the file
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS demo (id INTEGER PRIMARY KEY, name TEXT)")
cursor.execute("INSERT INTO demo (name) VALUES (?)", ("hello",))
conn.commit()
cursor.execute("SELECT * FROM demo")
print(cursor.fetchall())              # [(1, 'hello')]
conn.close()
```

**Key things to notice:**
- `?` placeholders prevent SQL injection (NEVER do `f"INSERT ... '{name}'"`)
- You must call `conn.commit()` to save changes
- `CREATE TABLE IF NOT EXISTS` is safe to call repeatedly

### What `db.py` needs to expose

Six functions that the rest of the codebase will call:

| Function | What it does |
|----------|-------------|
| `get_conn()` | Returns a connection to the SQLite database. Creates the `data/` directory if it doesn't exist. |
| `init_db()` | Creates all 3 tables (`files`, `quarantine_log`, `scan_runs`) if they don't exist. Call this once at startup. |
| `upsert_file(record: FileRecord)` | Insert a new file row, or update it if the path already exists. This is used by the scanner. |
| `update_columns(file_id: int, **cols)` | Update specific columns for a file by its ID. Example: `update_columns(42, sha256="abc123", exact_group=3)`. Used everywhere. |
| `fetch_df(query: str, params=())` | Run a SELECT query and return a `pandas.DataFrame`. Used by the UI and ML feature builder. |
| `files_by_kind(kind: str)` | Shortcut: return all active files of a given kind (e.g., "image"). |

### Your blueprint

```python
# db.py — SQLite database connection, schema, and helper queries

import sqlite3
from pathlib import Path
import pandas as pd
from config import DB_PATH
from models import FileRecord


def get_conn() -> sqlite3.Connection:
    """Return a connection to the SQLite database.
    
    Create the parent directory if it doesn't exist.
    Enable row_factory = sqlite3.Row so you can access columns by name.
    """
    # TODO: 
    # 1. Make sure the parent directory of DB_PATH exists (Path.mkdir)
    # 2. Connect to the database
    # 3. Set row_factory so results come back as dict-like rows
    # 4. Return the connection
    pass


def init_db():
    """Create all tables if they don't exist.
    
    Copy the CREATE TABLE statements from the spec (Section 5).
    Call conn.commit() after executing them.
    """
    # TODO: Execute the 3 CREATE TABLE statements
    pass


def upsert_file(record: FileRecord):
    """Insert a file record, or update it if the path already exists.
    
    Hint: SQLite has INSERT OR REPLACE, but be careful — it deletes
    and re-inserts the row (losing the old ID). A safer pattern is:
      INSERT ... ON CONFLICT(path) DO UPDATE SET ...
    
    Think about which columns to set on insert vs update.
    """
    # TODO: Build the SQL, extract values from the FileRecord
    pass


def update_columns(file_id: int, **cols):
    """Update specific columns for a file.
    
    Usage: update_columns(42, sha256="abc", exact_group=3)
    
    Hint: Build the SET clause dynamically from the kwargs.
      SET col1 = ?, col2 = ?
    """
    # TODO: 
    # 1. Build "SET col1=?, col2=?" from the keys
    # 2. Execute UPDATE files SET ... WHERE id = ?
    # 3. Commit
    pass


def fetch_df(query: str, params=()) -> pd.DataFrame:
    """Run a SELECT query and return results as a pandas DataFrame.
    
    Hint: pandas has pd.read_sql_query(query, conn, params=params)
    """
    # TODO: One-liner with pd.read_sql_query
    pass


def files_by_kind(kind: str) -> pd.DataFrame:
    """Return all active files of a given kind."""
    # TODO: Use fetch_df with a WHERE clause
    pass
```

### Key decisions to think about

- **Connection management:** Should you open one connection and keep it? Or open/close per function call? For this project, opening per call is simpler and avoids threading issues with Streamlit. You can optimize later if needed.
- **`sqlite3.Row` as `row_factory`:** This lets you do `row["name"]` instead of `row[1]`. Much more readable.
- **`INSERT ... ON CONFLICT(path) DO UPDATE`:** This is the SQLite "upsert" pattern. If a file was already scanned, you want to update its metadata, not create a duplicate row.
- **`**kwargs` for `update_columns`:** This is a Python power feature. The function receives arbitrary keyword arguments as a dictionary. You build the SQL dynamically from the keys.

### How to test it

```python
from db import init_db, upsert_file, update_columns, fetch_df
from models import FileRecord

# Initialize the database
init_db()

# Insert a test record
rec = FileRecord(
    path="C:/test/photo.jpg",
    name="photo.jpg", ext="jpg", kind="image",
    size=50000, created=1700000000.0,
    modified=1700000000.0, accessed=1700000000.0,
)
upsert_file(rec)

# Query it back
df = fetch_df("SELECT * FROM files WHERE kind = ?", ("image",))
print(df)
print(f"Columns: {list(df.columns)}")

# Update a field
file_id = df.iloc[0]["id"]
update_columns(file_id, sha256="abc123test", exact_group=1)

# Verify the update
df2 = fetch_df("SELECT sha256, exact_group FROM files WHERE id = ?", (file_id,))
print(df2)
```

Delete `data/clutteriq.db` between test runs to start fresh.

### ✅ When you're done, show me your `db.py` and I'll review it.

---

## The Workflow

```
1. Write config.py     →  test imports in Python shell
2. Write models.py     →  test creating FileRecord objects
3. Write db.py         →  test init_db, upsert, query
```

**Start with `config.py`** — it's the simplest and everything imports from it. Once it works, move to `models.py`, then `db.py`.

Show me each file as you finish it and I'll review your code, point out issues, and help you understand anything that's tricky. Take your time — these 3 files are the foundation everything else builds on. 🧱
