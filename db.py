"""SQLite database interface for ClutterIQ.

Provides connection management, schema initialization, and high-level
CRUD helpers so pipeline modules do not write raw SQL statements directly.

Both Person A and Person B share this contract to access `data/clutteriq.db`.
"""

from pathlib import Path
import sqlite3
from typing import Any, List, Optional
import pandas as pd

from config import DB_PATH
from models import FileRecord

# ---------------------------------------------------------------------------
# Schema definitions (Shared Data Contract)
# ---------------------------------------------------------------------------
SCHEMA_FILES = """
CREATE TABLE IF NOT EXISTS files (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  path            TEXT UNIQUE NOT NULL,   -- absolute path
  name            TEXT,
  ext             TEXT,                   -- lowercase, no dot
  kind            TEXT,                   -- image|document|video|audio|archive|code|other
  size            INTEGER,                -- bytes
  created         REAL,                   -- epoch seconds
  modified        REAL,
  accessed        REAL,                   -- NOTE: unreliable on some OS; treat as a weak signal
  sha256          TEXT,                   -- filled only for size-collision candidates
  phash           TEXT,                   -- hex string, images only
  width           INTEGER,
  height          INTEGER,
  blur_score      REAL,                   -- Laplacian variance; higher = sharper
  emb_index       INTEGER,                -- row index into data/embeddings.npy
  exact_group     INTEGER,                -- group id for exact duplicates, else NULL
  near_group      INTEGER,                -- group id for pHash near-duplicates, else NULL
  semantic_group  INTEGER,                -- group id for CLIP clusters, else NULL
  is_keeper       INTEGER DEFAULT 0,      -- 1 if best copy in its group
  category        TEXT,                   -- optional zero-shot label (screenshot, meme, receipt...)
  junk_score      REAL,                   -- 0..1 probability file is safe to remove
  junk_reason     TEXT,                   -- short human explanation
  privacy_type    TEXT,                   -- aadhaar|pan|card|bank_statement|NULL
  privacy_masked  TEXT,                   -- masked sample only, never the full value
  user_label      TEXT,                   -- keep|delete|NULL (from labelling page)
  status          TEXT DEFAULT 'active'   -- active|quarantined|restored
);
"""

SCHEMA_QUARANTINE_LOG = """
CREATE TABLE IF NOT EXISTS quarantine_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id         INTEGER,
  original_path   TEXT,
  quarantine_path TEXT,
  sha256          TEXT,
  moved_at        REAL,
  restored_at     REAL
);
"""

SCHEMA_SCAN_RUNS = """
CREATE TABLE IF NOT EXISTS scan_runs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  root            TEXT,
  started         REAL,
  finished        REAL,
  files_seen      INTEGER
);
"""

# Tuple of all column names in the exact order FileRecord.from_row expects
SELECT_COLUMNS_STR = (
    "id, path, name, ext, kind, size, created, modified, accessed, "
    "sha256, phash, width, height, blur_score, emb_index, exact_group, "
    "near_group, semantic_group, is_keeper, category, junk_score, "
    "junk_reason, privacy_type, privacy_masked, user_label, status"
)


# ---------------------------------------------------------------------------
# Connection & Initialization
# ---------------------------------------------------------------------------
def get_conn(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Return an open SQLite connection, ensuring the parent directory exists.

    Enables foreign keys and WAL mode for smooth concurrent reads.
    Note: The caller is responsible for closing this connection when finished.
    """
    path_obj = Path(db_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path_obj), timeout=30.0)
    # WAL (Write-Ahead Logging) enables concurrent readers while writing
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Create all database tables and performance indexes if they do not exist."""
    conn = get_conn(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(SCHEMA_FILES)
            cursor.execute(SCHEMA_QUARANTINE_LOG)
            cursor.execute(SCHEMA_SCAN_RUNS)

            # Performance indexes for frequent queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_kind ON files(kind);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_exact_group ON files(exact_group);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_near_group ON files(near_group);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_semantic_group ON files(semantic_group);")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# High-Level Record Operations
# ---------------------------------------------------------------------------
def upsert_file(record: FileRecord, db_path: Path | str = DB_PATH) -> int:
    """Insert a new file record or update metadata if path already exists.

    Returns:
        int: The database ID (primary key) of the inserted or existing row.
    """
    fields = record.to_dict()
    col_names = list(fields.keys())
    placeholders = ", ".join(["?"] * len(col_names))
    columns_str = ", ".join(col_names)

    # If duplicate path, update filesystem metadata and preserve analysis groups
    update_assignments = ", ".join([
        f"{col} = excluded.{col}"
        for col in col_names
        if col not in ("exact_group", "near_group", "semantic_group", "is_keeper", "user_label")
    ])

    sql = f"""
    INSERT INTO files ({columns_str})
    VALUES ({placeholders})
    ON CONFLICT(path) DO UPDATE SET
        {update_assignments}
    RETURNING id;
    """

    values = [fields[col] for col in col_names]
    conn = get_conn(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)
            row = cursor.fetchone()
            if row:
                record.id = row[0]
                return row[0]

            # Fallback query if RETURNING is not supported in legacy sqlite builds
            cursor.execute("SELECT id FROM files WHERE path = ?", (record.path,))
            fetched = cursor.fetchone()
            row_id = fetched[0] if fetched else 0
            record.id = row_id
            return row_id
    finally:
        conn.close()


def update_columns(file_id: int, db_path: Path | str = DB_PATH, **cols: Any) -> None:
    """Update one or more specific columns on a single file by ID.

    Example:
        update_columns(42, semantic_group=3, emb_index=12)
    """
    if not cols:
        return

    set_clause = ", ".join([f"{col} = ?" for col in cols.keys()])
    values = list(cols.values())
    values.append(file_id)

    sql = f"UPDATE files SET {set_clause} WHERE id = ?"
    conn = get_conn(db_path)
    try:
        with conn:
            conn.execute(sql, values)
    finally:
        conn.close()


def get_file_by_id(file_id: int, db_path: Path | str = DB_PATH) -> Optional[FileRecord]:
    """Retrieve a single FileRecord by primary key ID, or None if not found."""
    sql = f"SELECT {SELECT_COLUMNS_STR} FROM files WHERE id = ?"
    conn = get_conn(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (file_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return FileRecord.from_row(row)
    finally:
        conn.close()


def get_file_by_path(path: str, db_path: Path | str = DB_PATH) -> Optional[FileRecord]:
    """Retrieve a single FileRecord by absolute path, or None if not found."""
    sql = f"SELECT {SELECT_COLUMNS_STR} FROM files WHERE path = ?"
    conn = get_conn(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (str(path),))
        row = cursor.fetchone()
        if not row:
            return None
        return FileRecord.from_row(row)
    finally:
        conn.close()


def files_by_kind(
    kind: str,
    status: str = "active",
    db_path: Path | str = DB_PATH
) -> List[FileRecord]:
    """Return all active FileRecords matching a specific kind (e.g., 'image')."""
    sql = f"SELECT {SELECT_COLUMNS_STR} FROM files WHERE kind = ? AND status = ?"
    conn = get_conn(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (kind.lower(), status.lower()))
        rows = cursor.fetchall()
        return [FileRecord.from_row(r) for r in rows]
    finally:
        conn.close()


def get_active_files(db_path: Path | str = DB_PATH) -> List[FileRecord]:
    """Return all files whose status is 'active'."""
    sql = f"SELECT {SELECT_COLUMNS_STR} FROM files WHERE status = 'active'"
    conn = get_conn(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return [FileRecord.from_row(r) for r in rows]
    finally:
        conn.close()


def fetch_df(
    query: str,
    params: tuple = (),
    db_path: Path | str = DB_PATH
) -> pd.DataFrame:
    """Execute any SELECT query and return the result as a pandas DataFrame."""
    conn = get_conn(db_path)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
