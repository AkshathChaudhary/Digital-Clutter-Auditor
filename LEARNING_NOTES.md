# ClutterIQ: Learning Notes & Concept Glossary

This document serves as your personal revision guide and viva preparation notes. As we build each file, key concepts, design trade-offs, and self-quizzes are recorded here.

---

## Day 1: System Foundations & The Shared Data Contract (`models.py`, `config.py`, `db.py`)

### 1. Concepts Glossary

#### Dataclasses & Memory Optimization (`dataclasses`, `slots=True`)
- **What it is**: In standard Python classes, object attributes are stored in a dynamic internal dictionary (`__dict__`). While flexible, this adds significant memory overhead when dealing with tens of thousands of file records.
- **Why we use `slots=True`**: `slots` tells Python to allocate a fixed chunk of memory for the specified fields instead of creating `__dict__`. This reduces memory usage by up to 40-50% and speeds up attribute access.

#### Python `sqlite3` Connection vs. Transaction Context Managers
- **The Gotcha**: In Python's `sqlite3`, writing `with conn:` manages a **transaction** (it commits on block completion and rolls back on exception). However, it **does NOT close the connection**!
- **Windows File Locking (`[WinError 32]`)**: On Windows NTFS, if a file handle remains open, the operating system strictly forbids deleting or moving the file. Every connection opened in a helper function must be explicitly closed in a `finally: conn.close()` block.

#### WAL Mode (Write-Ahead Logging in SQLite)
- **What it is**: Standard SQLite locks the entire database file during writes, blocking any concurrent readers. With `PRAGMA journal_mode=WAL;`, changes are appended to a separate `-wal` file.
- **Why we use it**: It allows the Streamlit dashboard to read and display charts while the scanner or ML model is writing updates in the background without `database is locked` errors.

#### Parameterized Queries (Preventing SQL Injection & Syntax Bugs)
- **What it is**: Never format SQL queries with f-strings like `f"WHERE path = '{path}'"` because filenames containing single quotes (e.g., `O'Reilly_Book.pdf`) will crash the SQL parser or open injection vulnerabilities.
- **Why we use it**: Using `?` placeholders (e.g., `WHERE path = ?`, `(path,)`) allows SQLite to safely treat the parameter as pure data, regardless of quotes or special characters.

#### Upsert Pattern (`INSERT ... ON CONFLICT DO UPDATE`)
- **What it is**: "Upsert" is a combination of *Update* and *Insert*. If a row with a given `path` does not exist, SQLite inserts it. If it already exists (due to `path UNIQUE`), it updates the modified time/size while preserving previously computed ML scores and groups.

---

### 2. Day 1 Self-Quiz (5 Questions)

1. **Why do we store only absolute paths in `files.path` and enforce a `UNIQUE` constraint?**
   - *Answer*: Relative paths depend on the current working directory from which a script was launched. An absolute path uniquely identifies a file on disk, preventing the same file from being indexed multiple times.

2. **Why does `db.py` use `fetch_df()` returning a pandas DataFrame instead of raw tuples everywhere?**
   - *Answer*: Streamlit and Plotly work natively with pandas DataFrames. Returning DataFrames directly simplifies building aggregations, charts, and treemaps without writing manual loops.

3. **What is the purpose of `RETURNING id` in the `upsert_file` query?**
   - *Answer*: It immediately gives us the primary key of the row that was just inserted or updated in a single atomic SQL call, avoiding an extra `SELECT` round-trip.

4. **Why are database indices added on `path`, `kind`, `status`, and group columns?**
   - *Answer*: Without indexes, queries like `WHERE kind = 'image'` or `WHERE exact_group = 5` perform a full table scan ($O(N)$). Indexes create B-trees that allow lookups in $O(\log N)$ time.

5. **Why do we never delete records from the `files` table when a file is quarantined?**
   - *Answer*: We update `status = 'quarantined'`. Keeping historical records preserves the audit trail, duplicate group context, and enables instant one-click restoration.
