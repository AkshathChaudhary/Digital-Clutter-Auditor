# ClutterIQ: Project Specification (give this whole file to your AI coding tool)

> **Read this first, AI assistant.** This file is the complete specification for a college project built by a team of two students in **ONE WEEK**. Build exactly what is described here, in the order given. Do not skip sections. Do not invent extra features before the MUST-HAVE list is finished. The students are learning Python and ML, so **you must follow the "Teaching Mode" rules in Section 2 for every file you create.**

---

## 0. One-paragraph summary

**ClutterIQ** is a local-first, privacy-aware "digital hygiene auditor". It scans a folder (photos, downloads, a Google Drive/Takeout export), and then:
1. finds **exact duplicates** and **near-duplicate / similar images**;
2. picks the **best copy to keep** in each group (sharpest, highest resolution);
3. scores every file for **junk / staleness** with a small ML model that learns from the user's Keep/Delete feedback;
4. detects **sensitive documents** (Aadhaar, PAN, bank statements, card numbers) lying around, with values always **masked**;
5. shows everything in a **Streamlit dashboard** (space recoverable in GB, duplicate groups with thumbnails, privacy risks, ranked suggestions with reasons);
6. never deletes anything permanently: files go to a **quarantine folder with a manifest** and can be **restored in one click**;
7. is **measured**: a synthetic benchmark with known ground truth gives precision / recall / F1.

Everything runs 100% on the user's machine. No file ever leaves the computer.

**Why it is unique:** most class projects are classifiers on public datasets. This works on real messy personal data, adds a privacy angle, is explainable, is measured, and is easy to demo.

---

## 1. Team, constraints and rules

- **Team:** 2 students. **Person A** = Engine & Safety. **Person B** = Intelligence & Dashboard. (Details in Section 7.)
- **Time:** 7 days total. Day 7 is for report, slides and demo rehearsal only (code freeze at midday of Day 7).
- **Hardware assumption:** ordinary laptop, CPU only, no GPU. Python 3.11.
- **OS:** may be Windows, Linux or macOS. Use `pathlib` everywhere. Never hard-code `/` or `\`.
- **Safety rule (non-negotiable):** the tool is **read-only by default**. It never calls `os.remove`, `Path.unlink`, or `shutil.rmtree` on user files. The only way a file leaves its place is `shutil.move` into the quarantine folder, recorded in a manifest, restorable.
- **Test rule:** all development and testing is done on **copies** of folders or on generated benchmark data, never on real irreplaceable data.
- **Privacy rule:** sensitive values (Aadhaar, PAN, card numbers) are **never stored or displayed in full**. Store only the masked form (for example `XXXX XXXX 1234`) and the type of match.
- **Local-only rule:** no network calls at runtime (except the one-time download of the CLIP model weights during setup).

---

## 2. TEACHING MODE (mandatory behaviour for the AI assistant)

The students want to **learn every Python file** you create, not just receive them. Follow these rules throughout:

1. **Build one file (or one small group of related files) at a time**, in the order of Section 9 (daily plan). Do not dump the whole codebase at once.
2. For **every file** you create, immediately after the code, give a **"File Walkthrough"** containing:
   - **Purpose:** one or two sentences on why this file exists and where it fits in the pipeline.
   - **Top-to-bottom walkthrough:** explain each function and each important block in plain English, in the order they appear. Explain *why* the code is written that way, not only *what* it does.
   - **Key concepts:** name the Python or ML concepts used (for example generators, dataclasses, context managers, hashing, Hamming distance, embeddings, cosine similarity, train/test split) with a 2-3 line beginner explanation each.
   - **Libraries used:** why each library was chosen and what its main calls do.
   - **How it connects:** which other files call this one and which files it calls, and what data it reads or writes in the SQLite database.
   - **How to run and test it:** exact command(s) and what output to expect.
   - **Try-it exercise:** one small change the student can make (for example "change the Hamming threshold from 6 to 10 and see how the groups change") and what they should observe.
   - **3 viva questions with short model answers** about that file.
3. Use **short, clear comments in the code** for non-obvious lines, and **docstrings** on every function (purpose, arguments, return value).
4. Keep the code **simple and readable** over clever. Prefer plain functions and `dataclasses` over deep class hierarchies. Use type hints.
5. After finishing each day's work, **update `LEARNING_NOTES.md`** in the repo: a running glossary of concepts introduced that day plus a 5-question self-quiz.
6. If a student says "explain more" or "I don't understand X", slow down, use an analogy, and show a tiny standalone example.
7. After each file, **tell the student how to verify it works** before moving on. Do not proceed to the next file until they confirm.
8. At the end, produce a **"Whole-System Walkthrough"**: how one file travels through scan -> dedup -> quality -> ML -> privacy -> UI -> quarantine, so both students can explain the full system in the viva.
9. Be honest about limitations in the code and in the explanations. If something is a heuristic, say so.

---

## 3. Tech stack

| Area | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Required by course, rich ML libraries |
| Scanning | `pathlib`, `os`, `datetime` | Standard library, cross-platform |
| Hashing | `hashlib` (SHA-256) | Reliable exact-duplicate confirmation |
| Images | `Pillow`, `opencv-python-headless`, `imagehash` | Load images, blur score, perceptual hash |
| Semantic similarity | `sentence-transformers` with model `clip-ViT-B-32` | Image embeddings on CPU, simple API |
| Nearest neighbours | `scikit-learn` `NearestNeighbors` (cosine) | No extra install; fine for a few thousand images |
| ML model | `scikit-learn` `RandomForestClassifier` | Small tabular data, explainable, trains in seconds |
| Explainability | `permutation_importance` plus per-file rule reasons | Simple and explainable in a viva |
| OCR | `pytesseract` (needs Tesseract installed) | Reads text from images |
| PDF text | `pypdf` | Extract text from PDFs without OCR |
| Database | `sqlite3` (standard library) | One shared file, no server |
| UI | `streamlit`, `plotly`, `pandas` | Fast dashboard with charts |
| Tests | `pytest` | Unit tests |
| Progress bars | `tqdm` | Feedback during long scans |
| Version control | Git and GitHub | Two people, one repo |

### requirements.txt

```
pillow
opencv-python-headless
imagehash
numpy
pandas
scikit-learn
sentence-transformers
torch
pytesseract
pypdf
streamlit
plotly
tqdm
pytest
joblib
```

Setup notes the AI must give the students: install Python 3.11, create a virtual environment (`python -m venv .venv`), activate it, `pip install -r requirements.txt`, install the **Tesseract OCR** program separately (Windows installer or `apt install tesseract-ocr`) and make sure `pytesseract` can find it. The CLIP model downloads once (about 600 MB) on first use, so do this on good internet on Day 1.

---

## 4. Repository structure

```
clutteriq/
  README.md
  LEARNING_NOTES.md
  requirements.txt
  config.py                 # all thresholds and paths in one place
  main.py                   # CLI entry point: scan / analyze / app
  db.py                     # SQLite connection + schema + helper queries   (SHARED)
  models.py                 # dataclasses (FileRecord etc.)                  (SHARED)
  scanner/
    __init__.py
    walker.py               # A: folder walk + metadata
  dedup/
    __init__.py
    exact.py                # A: size -> SHA-256 grouping
    perceptual.py           # A: pHash near-duplicates
    semantic.py             # B: CLIP embeddings + clusters
  quality/
    __init__.py
    blur_rank.py            # A: blur score, resolution, keeper choice
  ml/
    __init__.py
    features.py             # B: feature table builder
    junk_model.py           # B: train / predict / explain / retrain
  privacy/
    __init__.py
    sensitive.py            # B: OCR + regex + masking
  safety/
    __init__.py
    quarantine.py           # A: move, manifest, restore
  bench/
    __init__.py
    make_benchmark.py       # A: synthetic dataset with ground truth
    evaluate.py             # B: precision / recall / F1 and charts
  app/
    app.py                  # Streamlit entry (page router)
    page_overview.py        # B
    page_duplicates.py      # A
    page_privacy.py         # B
    page_suggestions.py     # B (ranked list + Keep/Delete labelling)
    page_quarantine.py      # A
  tests/
    test_walker.py
    test_exact.py
    test_perceptual.py
    test_quarantine.py
    test_sensitive.py
    test_features.py
  data/                     # git-ignored: clutteriq.db, embeddings.npy, model.joblib
  quarantine/               # git-ignored
```

---

## 5. Shared data contract (the single most important part)

Both students read and write **one SQLite database** at `data/clutteriq.db`. Agree on this schema on Day 1 and **do not change column names without telling the other person**.

```sql
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

CREATE TABLE IF NOT EXISTS quarantine_log (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id       INTEGER,
  original_path TEXT,
  quarantine_path TEXT,
  sha256        TEXT,
  moved_at      REAL,
  restored_at   REAL
);

CREATE TABLE IF NOT EXISTS scan_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  root TEXT, started REAL, finished REAL, files_seen INTEGER
);
```

`db.py` must expose small helpers so nobody writes raw SQL in many places: `get_conn()`, `init_db()`, `upsert_file(record)`, `update_columns(file_id, **cols)`, `fetch_df(query, params)` (returns pandas DataFrame), `files_by_kind(kind)`.

`models.py` defines a `FileRecord` dataclass mirroring the columns above.

`config.py` holds constants:

```python
DB_PATH = "data/clutteriq.db"
EMB_PATH = "data/embeddings.npy"
MODEL_PATH = "data/junk_model.joblib"
QUARANTINE_DIR = "quarantine"
IMAGE_EXTS = {"jpg","jpeg","png","webp","bmp","gif","tiff"}
DOC_EXTS = {"pdf","doc","docx","txt","xlsx","pptx","csv"}
PHASH_MAX_DISTANCE = 6        # Hamming distance for near-duplicate (tune on benchmark)
CLIP_SIM_THRESHOLD = 0.92     # cosine similarity for semantic grouping (tune on benchmark)
BLUR_THRESHOLD = 100.0        # below this, an image counts as blurry
STALE_DAYS = 365
MIN_SIZE_FOR_HASH = 1         # bytes
CLIP_BATCH_SIZE = 32
```

---

## 6. Module specifications (what each file must do)

### 6.1 `scanner/walker.py` (Person A)
- `scan_folder(root: str) -> int`: walk `root` recursively with `pathlib.Path.rglob` or `os.walk`. For each regular file collect name, ext, size, created/modified/accessed times, and `kind` (from extension sets in `config.py`). Insert or update rows in `files`. Skip symlinks, skip unreadable files with a logged warning (never crash on `PermissionError`). Record a row in `scan_runs`. Show a `tqdm` bar. Return the number of files seen.
- Must never read file contents here (metadata only), so it is fast.
- `main.py scan --root <folder>` calls this.

### 6.2 `dedup/exact.py` (Person A)
- `find_exact_duplicates()`: group `active` files by `size`; only for sizes that occur more than once compute SHA-256 (read in 1 MB chunks, not all at once); store `sha256`; files with equal hash get the same `exact_group` id. Skip zero-byte files in grouping (report them separately as "empty files").
- Explain in the walkthrough *why size-first is a big speed-up*.

### 6.3 `dedup/perceptual.py` (Person A)
- `compute_phashes()`: for each image row without a `phash`, open with Pillow (catch corrupt files), compute `imagehash.phash`, store hex string and width/height.
- `find_near_duplicates()`: compare pHashes using Hamming distance <= `PHASH_MAX_DISTANCE`; cluster with union-find (connected components) so that A~B and B~C end up in one group; write `near_group`. Do not put files that are already byte-identical in a separate near group from their exact group: treat exact duplicates as one node.
- Complexity note: O(n^2) pairwise is acceptable for a few thousand images; mention BK-tree or bucketing as a "future work" idea only.

### 6.4 `quality/blur_rank.py` (Person A)
- `compute_blur(path) -> float`: OpenCV grayscale, `cv2.Laplacian(gray, cv2.CV_64F).var()`.
- `choose_keepers()`: for each duplicate group (exact, near, semantic), pick the keeper using a score: highest resolution first, then highest blur score (sharpest), then largest file size, then oldest modified date as final tie-break. Set `is_keeper = 1` for the winner, 0 for the rest. Save the reason string (for example "sharpest and highest resolution") in `junk_reason` for the non-keepers.
- Files not in any group are not touched here.

### 6.5 `safety/quarantine.py` (Person A)
- `quarantine_files(file_ids: list[int]) -> int`: for each id, verify the file still exists, compute/confirm SHA-256, `shutil.move` it into `quarantine/<timestamp>/` keeping a unique name, write a row in `quarantine_log`, set `status='quarantined'`. Write/update `quarantine/manifest.json` as a human-readable backup of the log.
- `restore_files(file_ids: list[int]) -> int`: move back to `original_path` (recreate parent folders; if the original path is occupied, restore with a `_restored` suffix and warn), verify the hash matches, set `status='restored'`, set `restored_at`.
- `list_quarantined()` returns a DataFrame.
- **No function in the project may permanently delete user files.** If an "empty quarantine" feature is added, it must be behind a typed confirmation and must be Day-7-optional.

### 6.6 `bench/make_benchmark.py` (Person A)
- `make_benchmark(src_images_dir, out_dir, n_base=60)`: takes a folder of clean images (the student can download any free photo set, or use their own copies) and creates a messy test folder with **known ground truth** saved to `bench/ground_truth.json`:
  - exact copies (renamed, in different subfolders);
  - near-duplicates: resized, JPEG-recompressed at lower quality, slightly cropped, brightness changed, light blur;
  - unrelated "distractor" images;
  - fake sensitive documents: generated images or text files containing **fake** numbers (clearly fake Aadhaar-format and PAN-format values) rendered as text with Pillow, plus a few look-alike non-sensitive numbers as negatives;
  - some junk: tiny files, `Copy of ...`, `(1)` names, old modification dates (set with `os.utime`).
- The ground truth maps each file to its group id and labels (duplicate_of, sensitive, junk).

### 6.7 `dedup/semantic.py` (Person B)
- `compute_embeddings()`: load `SentenceTransformer('clip-ViT-B-32')`; embed images in batches of `CLIP_BATCH_SIZE`; L2-normalise; save the array to `EMB_PATH`; store each file's row index in `emb_index`. **Cache**: skip images already embedded, and do not recompute on re-runs.
- `cluster_semantic()`: use `NearestNeighbors(metric='cosine')` or a similarity matrix to connect images with cosine similarity >= `CLIP_SIM_THRESHOLD`; union-find into `semantic_group`. Only keep groups of size >= 2.
- Optional (only if ahead of schedule): `tag_categories()` using CLIP zero-shot with text prompts ("a screenshot", "a meme", "a photo of a receipt", "a photo of a document", "a normal photograph") to fill `category`.
- Explain what an embedding is and why cosine similarity works.

### 6.8 `privacy/sensitive.py` (Person B)
- `scan_sensitive()`: for each image (Tesseract OCR), PDF (`pypdf` text; if no text, skip), and `.txt` file, extract text, then detect:
  - **Aadhaar-like:** 12 digits, first digit 2-9, optionally grouped as `XXXX XXXX XXXX` or `XXXX-XXXX-XXXX`;
  - **PAN-like:** regex `[A-Z]{5}[0-9]{4}[A-Z]`;
  - **Card-like:** 13-19 digits (with spaces/dashes) that pass the **Luhn** check;
  - **Bank statement keywords:** "account statement", "IFSC", "account number", "closing balance" (need at least 2 keywords).
- Store only `privacy_type` and `privacy_masked` (for example `XXXX XXXX 1234`, `ABCDE****F`). **Never store the raw match or the full OCR text.**
- Limit OCR cost: downscale large images before OCR, set a per-file timeout, and only OCR images that look like documents (optional pre-filter: skip very small images and obvious photos if CLIP category is available).
- Explain false positives honestly (a random 12-digit number can look like an Aadhaar number; the Verhoeff checksum can reduce this, implementing it is a nice extra).

### 6.9 `ml/features.py` (Person B)
- `build_feature_table() -> DataFrame`: one row per active file with:
  - `age_days` (now - modified), `days_since_access` (weak signal), `size_log` (log of size), `ext` (one-hot or label-encoded top extensions plus "other"), `kind`, `folder_depth`, `in_exact_group`, `in_near_group`, `in_semantic_group`, `is_keeper`, `is_blurry` (blur < threshold), `name_copy_pattern` (1 if name contains "copy", "(1)", "(2)", "final_final", "untitled"), `in_downloads_or_temp` (path contains downloads/temp/cache/tmp), `tiny_file` (size < 10 KB).
  - Never use `junk_score` or `user_label` as features.

### 6.10 `ml/junk_model.py` (Person B)
- **Cold start:** `heuristic_score(row) -> (score, reason)`: a transparent rule-based score from 0 to 1 (+0.3 non-keeper duplicate, +0.2 older than `STALE_DAYS`, +0.15 blurry, +0.15 copy-pattern name, +0.1 tiny, +0.1 in downloads/temp; capped at 1). Return a human-readable reason such as "duplicate, not accessed in 400 days, blurry". Use this until enough labels exist.
- **Labelling:** the Suggestions page lets users mark files `keep` or `delete`; stored in `user_label`. Target about 200-300 labelled files (use quick labelling over the benchmark data plus the students' own sample folders).
- `train_model()`: if >= 100 labelled rows with both classes, train a `RandomForestClassifier(n_estimators=200, class_weight='balanced')`, report 5-fold cross-validated accuracy/precision/recall, save with `joblib` to `MODEL_PATH`.
- `predict_scores()`: if a model exists, write `junk_score = predict_proba[:,1]`; otherwise fall back to `heuristic_score`. Always write a `junk_reason` made from the top contributing features (use feature importances plus the file's own feature values to generate the sentence).
- `retrain()`: called after the user corrects labels. Show before/after in the UI.
- Be explicit in the code comments and report: **labels come from the students' own judgement (weak supervision plus active learning)**. There is no public "safe to delete" dataset.

### 6.11 `bench/evaluate.py` (Person B)
- Load `bench/ground_truth.json`, run the pipeline on the benchmark folder, compare predicted groups against the truth:
  - duplicate detection: pairwise precision, recall, F1 for (a) exact only, (b) pHash only, (c) CLIP only, (d) pHash + CLIP combined;
  - privacy detection: precision/recall on the fake sensitive files and negatives;
  - junk model: cross-validated accuracy, confusion matrix;
  - runtime per 1000 files for each stage.
- Save results to `bench/results.csv` and charts (PNG) to `bench/figures/` for the report.

### 6.12 Streamlit app (`app/`)
- `app.py`: sidebar with navigation, folder path input, "Run scan", "Run analysis" buttons, and a progress indicator. Use `st.session_state` for selections. Cache DB reads with `st.cache_data` and clear the cache after writes.
- `page_overview.py` (B): big metric cards (total files, total size, **space recoverable in GB**, duplicate groups, privacy risks), Plotly treemap of space by folder and kind, bar chart of space by file type, "stale files" histogram by age.
- `page_duplicates.py` (A): list duplicate groups (exact, near, semantic) with thumbnails side by side; keeper highlighted with a green badge and the reason; checkboxes to select non-keepers; "Move selected to quarantine" button with a confirmation dialog.
- `page_privacy.py` (B): table of sensitive findings showing **only masked values**, file path, type; button to quarantine selected files; a short disclaimer explaining that detection is heuristic.
- `page_suggestions.py` (B): ranked list by `junk_score` with the reason and a Keep / Delete(quarantine) button per row; each click saves `user_label`; "Retrain model" button; show model metrics after retrain.
- `page_quarantine.py` (A): list quarantined files, "Restore selected" button, total quarantined size.
- **Space recoverable** = sum of sizes of non-keeper duplicates plus high-confidence junk (`junk_score >= 0.8`), without double-counting a file that is in several groups.

### 6.13 `main.py`
CLI using `argparse`:
```
python main.py scan --root <folder>
python main.py analyze            # exact -> phash -> blur/keepers -> clip -> privacy -> features -> junk scores
python main.py benchmark --src <clean_images_dir> --out <folder>
python main.py evaluate
streamlit run app/app.py
```

---

## 7. Division of labour

### Person A: Engine & Safety
Owns: `db.py`, `models.py`, `config.py` (with B reviewing), `scanner/`, `dedup/exact.py`, `dedup/perceptual.py`, `quality/`, `safety/`, `bench/make_benchmark.py`, `app/page_duplicates.py`, `app/page_quarantine.py`, `tests/test_walker.py`, `test_exact.py`, `test_perceptual.py`, `test_quarantine.py`, `main.py`.

### Person B: Intelligence & Dashboard
Owns: `dedup/semantic.py`, `privacy/`, `ml/`, `bench/evaluate.py`, `app/app.py`, `app/page_overview.py`, `app/page_privacy.py`, `app/page_suggestions.py`, `tests/test_sensitive.py`, `test_features.py`, and the evaluation charts.

### Shared
Day 1 schema agreement, repo setup, Day 6 integration, report, slides, demo, viva rehearsal. **Each person must be able to explain the other's half** (do a 15-minute swap-and-explain session on Day 6).

### Git workflow
One GitHub repo. `main` is always runnable. Each person works on a branch (`feature-a`, `feature-b`), commits small and often, opens a pull request, and the other person reviews and merges at least once a day. Add `data/`, `quarantine/`, `.venv/`, `__pycache__/` to `.gitignore`.

---

## 8. Priorities for a ONE-WEEK build

**MUST-HAVE (without these the project fails):**
1. Scanner + SQLite schema
2. Exact duplicates
3. pHash near-duplicates
4. Keeper selection (blur + resolution)
5. Quarantine + restore
6. Heuristic junk score with reasons
7. Streamlit pages: Overview, Duplicates, Quarantine
8. Benchmark generator + precision/recall for duplicates

**SHOULD-HAVE (the "unique" factor; do these next):**
9. CLIP semantic grouping
10. Privacy scanner with masking (Aadhaar, PAN, card with Luhn)
11. Trained RandomForest with labelling page and retrain button
12. Privacy and Suggestions pages

**NICE-TO-HAVE (only if everything above works):**
13. CLIP zero-shot categories
14. Verhoeff checksum for Aadhaar
15. Estimated storage cost saved in rupees using a **user-editable price per GB** (do not hard-code a vendor price)
16. Export a PDF/CSV clean-up report

**If behind schedule, cut in this order:** 15, 16, 14, 13, then simplify OCR to images only. Never cut the safety layer or the benchmark.

---

## 9. One-week plan (Day-by-day)

| Day | Person A | Person B | Together |
|---|---|---|---|
| **1** | Repo, venv, `db.py`, `models.py`, `config.py`, `scanner/walker.py`, `main.py scan` | Install and test CLIP and Tesseract on a few images; collect a sample photo set; write the `FileRecord` use in a tiny test | Agree on schema, create GitHub repo, branches, `.gitignore`. End of day: scan works on a sample folder |
| **2** | `dedup/exact.py`, `dedup/perceptual.py`, tests | `dedup/semantic.py` (embeddings + cache + clustering) | Quick merge; run both on the sample folder |
| **3** | `quality/blur_rank.py`, `safety/quarantine.py`, `test_quarantine.py` | `privacy/sensitive.py`, `test_sensitive.py` | Run `analyze` end to end on the sample folder |
| **4** | `bench/make_benchmark.py`, `app/page_duplicates.py`, `page_quarantine.py` | `ml/features.py`, `ml/junk_model.py` (heuristic first), `app/app.py`, `page_overview.py` | Generate benchmark folder and verify ground truth |
| **5** | Finish Duplicates and Quarantine UI, handle edge cases (corrupt images, permission errors, huge files), write missing tests | `page_privacy.py`, `page_suggestions.py` with labelling, label 200+ files, train model, `bench/evaluate.py` | **Integration checkpoint**: the whole app runs on the benchmark folder |
| **6** | Fix bugs, tune `PHASH_MAX_DISTANCE`, performance check, README | Tune `CLIP_SIM_THRESHOLD`, finalize evaluation tables and charts | **Swap-and-explain session**, freeze features at the end of the day |
| **7 (morning)** | Final bug fixes only; run a full demo rehearsal on a fresh copy of a real folder | Same | **CODE FREEZE at midday** |
| **7 (afternoon)** | Report (system design, safety) | Report (ML, evaluation), slides | Demo rehearsal x2, viva Q&A practice |

Daily rhythm: 10-minute stand-up at the start (what is done, what blocks you), pull requests reviewed before dinner, `LEARNING_NOTES.md` updated at the end of the day.

---

## 10. Testing checklist

- Scanner: handles empty folder, nested folders, unicode filenames, permission errors, symlinks.
- Exact: two identical files in different folders are grouped; files of equal size but different content are NOT grouped; zero-byte files handled.
- pHash: resized/recompressed copy is grouped; unrelated image is not; corrupt image does not crash the run.
- Keeper: sharper and larger copy wins; exactly one keeper per group.
- Quarantine: move then restore returns a byte-identical file (hash equal); restore when the original path is occupied does not overwrite anything.
- Privacy: known fake Aadhaar/PAN/card strings detected and masked; random 12-digit non-Aadhaar-looking numbers and invalid Luhn numbers are not flagged as cards; raw values never appear in the DB.
- ML: feature table has no leakage columns; model trains with >= 100 labels; falls back to heuristic when there is no model.
- UI: every page loads on an empty DB without errors and shows a friendly message.
- Run `pytest` before every pull request.

---

## 11. Demo script (about 5 minutes)

1. Explain the problem in one sentence: storage waste plus hidden sensitive files.
2. Run the scan on a prepared messy folder (a **copy**).
3. Show the Overview: "4.2 GB recoverable, 312 duplicate groups, 7 sensitive documents".
4. Open a duplicate group: keeper highlighted with the reason.
5. Open the Privacy page: masked Aadhaar/PAN hits, mention that nothing sensitive is stored in full.
6. In Suggestions, correct one wrong suggestion, press Retrain, show the ranking change.
7. Quarantine a file, then restore it.
8. Show the benchmark table (precision/recall for pHash vs CLIP vs both).
9. Close with limitations and future work.

---

## 12. Likely viva questions (prepare short answers)

- Why size first, then hash? (Skips hashing most files; only collisions need reading.)
- Why SHA-256 and not MD5? (Collision resistance; speed difference is minor here.)
- What is a perceptual hash and why Hamming distance? (Similar images give similar bit strings; distance counts differing bits.)
- Why both pHash and CLIP? (pHash is fast and catches edits; CLIP catches semantic similarity; the benchmark shows each one's gaps.)
- What is an embedding and cosine similarity?
- Why RandomForest and not deep learning? (Small tabular data, explainable, fast.)
- Where do your labels come from? (Our own judgement plus user feedback: weak supervision and active learning; there is no public dataset.)
- How do you avoid deleting important files? (Read-only default, quarantine, manifest, restore, user confirmation.)
- How do you protect privacy? (Local only, masked storage, no raw text saved.)
- What are the limitations? (Accessed time unreliable, OCR errors, Aadhaar regex false positives, CLIP slow on CPU, thresholds depend on data.)
- How did you validate? (Synthetic ground truth with known duplicates and sensitive files, plus cross-validation.)

---

## 13. Known risks and how to handle them

| Risk | Mitigation |
|---|---|
| CLIP is slow on CPU | Cache embeddings to `.npy`, batch of 32, resize images before embedding, test on <= 1000 images |
| Tesseract not found | Set `pytesseract.pytesseract.tesseract_cmd` in `config.py` with the install path |
| `accessed` times are not updated on many systems | Treat as a weak feature; rely on `modified` and `age_days` |
| Thresholds too loose/tight | Tune on the benchmark, document the chosen values and the chart |
| Streamlit reruns the script on each click | Use `st.session_state` and `st.cache_data`; clear the cache after DB writes |
| Merge conflicts in shared files | Only A edits `db.py`/`models.py`; B requests changes via a message; `config.py` edits are small and communicated |
| Running out of time | Follow the cut order in Section 8 |

---

## 14. Final deliverables

1. Working code in a GitHub repo with README (setup, usage, screenshots).
2. Benchmark results (`bench/results.csv` and figures).
3. `LEARNING_NOTES.md` with concept glossary and quizzes.
4. Project report: problem, related work (phone cleaners, `fdupes`, `dupeGuru`), design, algorithms, evaluation, limitations, ethics/privacy, future work.
5. Slides and a 5-minute demo.

---

## 15. First message to the AI (copy and paste this when you start)

> I have attached `CLUTTERIQ_PROJECT_SPEC.md`. Read the whole file carefully. We are a team of two with only one week. I am **Person [A/B]**. Follow Teaching Mode (Section 2): build one file at a time in the order of the Day plan, and after every file give me the File Walkthrough so I can learn it. Start with **Day 1**: first give me the exact setup steps, then create the files assigned to my role, then explain each one and tell me how to test it. Do not move to the next file until I say "next".
