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

---

## Day 2: Semantic Deduplication & Neural Embeddings (`dedup/semantic.py`)

### 1. Concepts Glossary

#### Vector Embedding (Latent Space Representation)
- **What it is**: An embedding is a fixed-length list of floating-point numbers (in our case, 512 dimensions) produced by a deep neural network that captures the *semantic meaning* of an image.
- **Why we use CLIP (`clip-ViT-B-32`)**: Traditional pixel matching fails if an image is cropped, recolored, or taken from a different angle. CLIP maps images with similar content to nearby points in a 512-dimensional vector space.

#### Python 3.12 + Windows SSL Recursion Bug (`truststore`)
- **The Gotcha**: On Windows, newer Hugging Face Hub releases route downloads through `httpx2` and `truststore` (to read the native Windows Certificate Store). Under Python 3.12, `truststore`'s internal `_set_ssl_context_verify_mode` triggers an infinite recursion loop via `super(SSLContext, SSLContext).verify_mode.__set__`.
- **The Resolution**: We created `ssl_fix.py` which patches the Python descriptors with direct calls to native C-level `_ssl._SSLContext.verify_mode.__set__`, bypassing the broken wrapper completely.

#### L2 Normalization (Unit Vectors)
- **What it is**: Scaling a vector so that its Euclidean length (norm) equals exactly 1:
  $$\mathbf{u}_{\text{norm}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2} = \frac{\mathbf{u}}{\sqrt{\sum_{i=1}^{d} u_i^2}}$$
- **Why we use it**: When vectors have unit length, the Cosine Similarity formula:
  $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
  simplifies directly to a basic **dot product** ($\mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}}$). This allows us to compute pairwise similarities across thousands of images in milliseconds using NumPy matrix multiplication (`X @ X.T`).

#### Disjoint Set Union (Union-Find) with Path Compression
- **What it is**: An efficient data structure that maintains a collection of disjoint sets.
- **Why we use it for clustering**: If image A is similar to image B ($\ge 0.92$), and image B is similar to image C ($\ge 0.92$), Union-Find unites them into a single transitive component $\{A, B, C\}$.

### 2. Day 2 Self-Quiz (5 Questions)

1. **Why do we need both perceptual hashing (pHash, Person A) and CLIP embeddings (Person B)?**
   - *Answer*: pHash is fast ($O(1)$) and catches exact crops/compression, but fails on semantic variations. CLIP catches semantic duplicates. Together they cover both exact visual edits and semantic duplicates.

2. **Why do we store embeddings in a separate `.npy` file instead of directly inside SQLite?**
   - *Answer*: Storing large floating-point arrays in SQLite blobs degrades database performance and prevents vectorized matrix operations. NumPy `.npy` files support lightning-fast memory-mapped linear algebra.

3. **What is the mathematical shortcut made possible by L2-normalizing CLIP embeddings?**
   - *Answer*: Cosine similarity simplifies to the matrix dot product $\mathbf{X} \mathbf{X}^T$, eliminating the need to divide by norms during pairwise comparisons.

4. **What does Path Compression in the Union-Find algorithm do?**
   - *Answer*: It re-points all visited nodes directly to the root during `find(x)`, preventing deep trees and keeping queries near $O(1)$.

5. **Why do solid color patches sometimes have high cosine similarity in CLIP?**
   - *Answer*: Vision Transformers divide images into patches and extract features. Flat, solid-color squares contain no edges, textures, or objects, so CLIP encodes them into the same conceptual category: "a blank flat monochrome canvas".

---

## Day 3: Privacy & Sensitive Document Detection (`privacy/sensitive.py`)

### 1. Concepts Glossary

#### The Luhn Algorithm (Modulus 10 Checksum)
- **What it is**: A simple checksum formula invented by Hans Peter Luhn in 1954 to validate a variety of identification numbers, particularly credit and debit card numbers (ISO/IEC 7812).
- **How it works**:
  1. From right to left, double every second digit.
  2. If doubling produces a two-digit number (e.g. $7 \times 2 = 14$), sum its digits or subtract 9 ($14 - 9 = 5$).
  3. Sum all final digits. If the total is divisible by 10 ($\text{sum} \pmod{10} == 0$), the card number is mathematically valid.
- **Why we use it**: It instantly filters out random 16-digit strings (like software licenses, timestamps, or tracking codes) from real credit/debit card numbers.

#### The Verhoeff Algorithm (Dihedral Group $D_5$ Checksum)
- **What it is**: A decimal check-digit algorithm developed in 1969 based on the symmetries of a regular pentagon (the non-abelian dihedral group $D_5$). It is the official checksum used in 12-digit Indian **Aadhaar** numbers.
- **Why it is superior to Luhn**: Luhn cannot detect transposition of adjacent digits like `27` $\rightarrow$ `72` or `09` $\rightarrow$ `90`. The Verhoeff algorithm detects **100% of single-digit substitution errors** and **100% of adjacent transposition errors**.

#### Privacy-by-Design & Zero-Persistence Masking
- **What it is**: A system architecture guarantee that sensitive personal identifiable information (PII) is destroyed in memory as soon as classification is finished.
- **Our Rule**: Full OCR transcripts and raw numbers are **never written to disk or the SQLite database**. Only the masked representation (`XXXX XXXX 1234`, `ABCDE****F`, `************1234`) is persisted.

#### Optical Character Recognition (OCR) Preprocessing & Optimization
- **Grayscale Conversion (`L`)**: Reduces image channels from 3 (RGB) to 1, cutting memory overhead by 66% and maximizing text-to-background contrast for Tesseract.
- **Adaptive Downscaling**: High-resolution mobile phone photos (e.g. 4000x3000) take 5–10 seconds to OCR on CPU. Downscaling to a maximum dimension of 1600px retains full text readability while speeding up OCR execution by 4x–5x.

---

### 2. Day 3 Self-Quiz (5 Questions)

1. **Why is regex alone insufficient for detecting credit card numbers?**
   - *Answer*: Any random 16-digit number (e.g., barcode, tracking number, product serial key) matches a 16-digit regex. Combining regex with the **Luhn algorithm** ensures only mathematically valid card numbers are flagged.

2. **Why does Aadhaar use the Verhoeff algorithm rather than a standard sum modulo 10?**
   - *Answer*: Humans frequently transpose two adjacent digits when typing (e.g. `12` instead of `21`). Verhoeff catches 100% of adjacent transpositions and 100% of single-digit typos using group theory ($D_5$).

3. **Why do we downscale high-resolution images to 1600px before running Tesseract OCR?**
   - *Answer*: Mobile photos are often 12+ megapixels. OCR on CPU scales with pixel count; downscaling to 1600px keeps text crisp while cutting processing time from 8 seconds down to ~1.5 seconds per image.

4. **What is our security guarantee regarding sensitive text and OCR transcripts?**
   - *Answer*: Raw text is processed in transient volatile RAM only; never logged, never sent to the network, and never written to SQLite. Only masked samples (`XXXX XXXX 1234`) and privacy flags are stored.

5. **How does our bank statement detector avoid false positives from emails or general invoices?**
   - *Answer*: It uses a keyword co-occurrence heuristic requiring at least **two distinct** primary financial terms (e.g. `IFSC` and `closing balance`), preventing single-word mentions from triggering an alert.
