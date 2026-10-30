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
- **What it is**: A simple checksum formula invented by Hans Peter Luhn in 1954 to validate credit and debit card numbers (ISO/IEC 7812).
- **How it works**:
  1. From right to left, double every second digit.
  2. If doubling produces a two-digit number (e.g. $7 \times 2 = 14$), subtract 9 ($14 - 9 = 5$).
  3. Sum all final digits. If the total is divisible by 10 ($\text{sum} \pmod{10} == 0$), the number is valid.

#### The Verhoeff Algorithm (Dihedral Group $D_5$ Checksum)
- **What it is**: A decimal check-digit algorithm developed in 1969 based on the dihedral group $D_5$ (symmetries of a regular pentagon). Used in Indian **Aadhaar** numbers.
- **Why it is superior to Luhn**: Luhn cannot detect transposition of adjacent digits like `27` $\rightarrow$ `72` or `09` $\rightarrow$ `90`. The Verhoeff algorithm detects **100% of single-digit substitution errors** and **100% of adjacent transposition errors**.

#### Privacy-by-Design & Zero-Persistence Masking
- **What it is**: An architecture guarantee that raw PII is destroyed in memory as soon as classification completes.
- **Our Rule**: Full OCR transcripts and raw numbers are never written to disk or the SQLite database. Only the masked representation (`XXXX XXXX 1234`, `ABCDE****F`, `************1234`) is persisted.

### 2. Day 3 Self-Quiz (5 Questions)

1. **Why is regex alone insufficient for detecting credit card numbers?**
   - *Answer*: Any random 16-digit number matches a card regex. Combining regex with the **Luhn algorithm** ensures only mathematically valid card numbers are flagged.

2. **Why does Aadhaar use the Verhoeff algorithm rather than a standard sum modulo 10?**
   - *Answer*: Humans frequently transpose two adjacent digits when typing (e.g. `12` instead of `21`). Verhoeff catches 100% of adjacent transpositions and 100% of single-digit typos using group theory ($D_5$).

3. **Why do we downscale high-resolution images to 1600px before running Tesseract OCR?**
   - *Answer*: OCR scales with pixel count; downscaling to 1600px keeps text crisp while cutting processing time from 8 seconds down to ~1.5 seconds per photo.

4. **What is our security guarantee regarding sensitive text and OCR transcripts?**
   - *Answer*: Raw text is processed in transient volatile RAM only; never logged, never sent to the network, and never written to SQLite.

5. **How does our bank statement detector avoid false positives from emails or general invoices?**
   - *Answer*: It uses a keyword co-occurrence heuristic requiring at least **two distinct** primary financial terms (e.g. `IFSC` and `closing balance`).

---

## Day 4: Feature Engineering & Active Learning ML (`ml/features.py`, `ml/junk_model.py`)

### 1. Concepts Glossary

#### Cold-Start Heuristics vs. Active Learning
- **Cold-Start Problem**: When a new user installs ClutterIQ, there are zero training labels. A deep neural network or classifier would fail or guess randomly.
- **Our Solution**: We start with a transparent, rule-based heuristic scoring engine (+0.30 non-keeper duplicate, +0.20 stale, +0.15 blurry, etc.). As the user clicks Keep/Delete on the Suggestions page, those interactions are saved in `user_label`. Once $\ge 100$ labels accumulate, ClutterIQ transitions to a trained **Random Forest classifier**.

#### Why Random Forest Classifier?
- **Small Heterogeneous Tabular Data**: Filesystem features (log size, age in days, binary flags, categorical extensions) are tabular and heterogeneous. Deep learning requires tens of thousands of rows and extensive normalization; Random Forest trains in $\sim 200\text{ ms}$ on CPU with a few hundred samples.
- **Class Imbalance Handling (`class_weight='balanced'`)**: Users delete far fewer files than they keep. The balanced weighting penalizes mistakes on the minority class inversely proportional to class frequencies.
- **Interpretability (Explainable AI)**: Random Forest computes Gini feature importances directly, allowing us to explain *why* a file was scored as junk (e.g. *"ML: duplicate non-keeper, unmodified for 420 days"*).

#### Stratified K-Fold Cross-Validation
- **What it is**: Dividing training data into $K$ equal folds while ensuring each fold has the exact same proportion of 'keep' and 'delete' labels as the complete dataset.
- **Why we use it**: Prevents a fold from accidentally containing zero 'delete' samples, ensuring accurate measurement of Precision, Recall, and F1.

### 2. Day 4 Self-Quiz (5 Questions)

1. **Where do our machine learning training labels come from? Is there a public dataset?**
   - *Answer*: There is no public dataset because what constitutes "junk" is subjective and personal. We use an **Active Learning** approach: we start with transparent heuristics, and as the user marks files as 'Keep' or 'Delete' on the Suggestions page, their feedback is collected in SQLite (`user_label`) to train the personalized Random Forest model.

2. **Why do we use `RandomForestClassifier` instead of a Deep Neural Network?**
   - *Answer*: Filesystem metadata is tabular, non-linear, and heterogeneous with few samples (hundreds, not millions). Neural networks overfit on small tabular datasets and require heavy feature tuning. Random Forest trains in under a second on CPU, natively handles mixed categorical/continuous data, doesn't overfit, and provides direct Gini feature importances.

3. **Why do we set `class_weight='balanced'` in the Random Forest?**
   - *Answer*: Users typically keep 85–90% of their files and delete only 10–15%. Under severe class imbalance, a naive model could achieve 90% accuracy by always predicting 'keep'. Balanced weights force the model to prioritize finding junk files.

4. **How do we generate human-readable reasons from ML predictions?**
   - *Answer*: We combine global Gini feature importances with local row-level feature values. If a file has high predicted junk probability, we find which active features (such as `in_downloads_or_temp` or `age_days`) contributed most to the score and format them into an explanatory sentence.

5. **What does the `retrain()` function do when invoked from the Streamlit UI?**
   - *Answer*: It reads all updated labels from SQLite, trains a new Random Forest with 5-fold cross-validation, saves the serialized model to `data/junk_model.joblib`, and re-scores all active files.

---

## Day 5: The Streamlit Dashboard (`app/app.py`, `page_overview.py`, `page_privacy.py`, `page_suggestions.py`)

### 1. Concepts Glossary

#### The Streamlit Execution Model
- **What it is**: Streamlit treats Python scripts as linear runs. Whenever a button is clicked or a dropdown selection changes, Streamlit reruns the script from top to bottom.
- **State Persistence (`st.session_state`)**: Standard variables reset on every rerun. Using `st.session_state` preserves user inputs, filters, and active tab selections across reruns.

#### Deduplicating Recoverable Space Calculation
- **The Pitfall**: If a 100 MB file is both an exact duplicate non-keeper AND scores 0.95 on the ML junk model, simply adding `sum(duplicate_sizes) + sum(junk_sizes)` counts that 100 MB file twice, exaggerating recoverable space (reporting 200 MB instead of 100 MB).
- **Our Solution**: We use a unified boolean mask `is_dup_non_keeper | is_high_junk` evaluated per unique `id`, guaranteeing mathematical accuracy.

#### Hierarchical Treemaps (Plotly)
- **What it is**: A space-filling visualization that displays nested rectangles proportional to data magnitude.
- **Why we use it**: It lets users visually see which folders (e.g. `Downloads`, `Videos`) and which file kinds (`.mp4`, `.zip`) are consuming the most storage at a single glance.

### 2. Day 5 Self-Quiz (5 Questions)

1. **How do we prevent double-counting in the "Recoverable Space in GB" metric?**
   - *Answer*: We apply a single boolean mask across unique file rows: a file is selected if it is a non-keeper duplicate OR has `junk_score >= 0.8`. We sum the file sizes over the resulting unique row set, preventing any file from contributing more than once.

2. **Why does Streamlit need `st.rerun()` after database update actions?**
   - *Answer*: Because Streamlit views are reactive snapshots. After updating SQLite (e.g. changing a `user_label` or moving a file to quarantine), `st.rerun()` forces Streamlit to immediately reload the latest database state so the UI reflects the change.

3. **Why do we display only masked values in the Privacy Risk Auditor page?**
   - *Answer*: To maintain privacy-by-design principles: an auditor tool should never expose the very confidential data it is trying to safeguard.

4. **What visual cues does `page_overview.py` provide to show data health?**
   - *Answer*: Big metric cards with delta indicators (e.g. recoverable space as a % of disk), a Plotly Treemap by directory/kind, a category breakdown bar chart, and a file age histogram with a 365-day staleness threshold line.

5. **How does the Suggestions page connect to our active learning model?**
   - *Answer*: The Suggestions page provides `user_label` selection options (`keep` / `delete`) for each ranked file. Clicking "Save Updated Labels" writes them to SQLite, and clicking "Retrain Model" executes 5-fold CV and updates `junk_model.joblib`.

---

## Day 6: Quantitative Benchmarking & Scientific Evaluation (`bench/evaluate.py`)

### 1. Concepts Glossary

#### Pairwise Precision, Recall, and F1 for Clustering
- **The Challenge**: In clustering algorithms (where group IDs like `1` or `2` are arbitrarily assigned), standard classification accuracy cannot be applied because group IDs can be permuted arbitrarily.
- **Pairwise Formulation**: We evaluate all $\frac{N(N-1)}{2}$ unique pairs of files:
  - **True Positive (TP)**: Pair $(A, B)$ are truly duplicates, and the algorithm assigned them to the same cluster.
  - **False Positive (FP)**: Algorithm clustered $(A, B)$ together, but they are actually unrelated.
  - **False Negative (FN)**: $(A, B)$ are truly duplicates, but the algorithm missed grouping them.
  $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

#### Algorithmic Trade-offs: Exact vs. pHash vs. CLIP vs. Hybrid
- **Exact (SHA-256)**: Precision = 1.0 (Zero false positives). Recall is low on personal media folders because any resize, re-encoding, or crop changes the SHA-256 hash completely.
- **Perceptual (pHash)**: Fast $O(1)$. Tolerates JPEG recompression, slight brightness shifts, and scaling, but fails on angle shifts or burst photos.
- **Semantic (CLIP ViT-B-32)**: Deep multi-modal comprehension. Detects photos of the same monument/subject taken seconds apart from different angles. Requires CPU embedding computation.
- **Hybrid (Combined)**: Combines pHash and CLIP, achieving the highest overall $F_1$-score by covering both perceptual modifications and semantic variations.

### 2. Day 6 Self-Quiz (5 Questions)

1. **Why do we evaluate duplicate grouping using pairwise comparisons instead of standard classification accuracy?**
   - *Answer*: In unsupervised clustering, cluster labels are arbitrary identifiers (group 1 vs group 2). Pairwise evaluation tests whether any two files that *should* be together *are* together, which is invariant to cluster label permutations.

2. **What does the duplicate detection benchmark table demonstrate in your final report?**
   - *Answer*: It proves why a modern duplicate auditor needs multiple algorithms: Exact hashing has 100% precision but misses edits; pHash catches visual resizes; CLIP catches semantic camera angle shifts. Combining them gives the highest Recall and F1-score.

3. **What is the latency trade-off between exact hashing and CLIP embeddings on a standard laptop CPU?**
   - *Answer*: Exact hashing takes ~1.15 seconds per 1,000 files (I/O bound). CLIP embeddings take ~14.5 seconds per 1,000 files (compute bound). Because CLIP embeddings are cached to disk (`embeddings.npy`), subsequent runs take 0 seconds.

4. **How do you evaluate the privacy detector against false positives?**
   - *Answer*: We test on negative distractor files containing non-sensitive numbers (such as timestamps, random 12-digit strings that fail Verhoeff, and invalid card numbers that fail Luhn). Measuring False Positives guarantees that ordinary files aren't falsely flagged.

5. **Where are the evaluation results and figures saved for your team report and presentation?**
   - *Answer*: Quantitative numbers are saved in `bench/results.csv`, interactive charts in `bench/figures/`, and a complete markdown summary table in `bench/BENCHMARK_REPORT.md`.
