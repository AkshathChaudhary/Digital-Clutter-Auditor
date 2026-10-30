# ClutterIQ: Privacy-Aware Local Digital Hygiene Auditor

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![100% Local](https://img.shields.io/badge/Privacy-100%25%20Local-green.svg)](#privacy-by-design)

**ClutterIQ** is a local-first, privacy-preserving digital auditor designed to reclaim gigabytes of storage and audit sensitive documents (Aadhaar, PAN, payment cards, bank statements) on personal computers.

All indexing, optical character recognition (OCR), multi-modal neural network embeddings, and machine learning scoring execute **100% locally on your machine**. No user file or personal data ever leaves your computer.

---

## Key Capabilities

1. **Multi-Modal Deduplication**:
   - **Exact Duplicates**: Chunked SHA-256 cryptographic hashing with size-first collision skipping.
   - **Perceptual Duplicates**: Image frequency discrete cosine transform (`pHash`) robust against JPEG re-compression, scaling, and crops.
   - **Semantic Duplicates**: OpenAI CLIP (`clip-ViT-B-32`) 512-dimensional vector embeddings with $O(1)$ disk caching (`embeddings.npy`) and cosine similarity Union-Find graph clustering.
2. **Intelligent Keeper Selection**:
   - Evaluates duplicate clusters using OpenCV Laplacian variance blur scores, resolution ($\text{width} \times \text{height}$), and file size to pick the sharpest, highest-quality keeper copy.
3. **Local Privacy Auditor**:
   - Scans images (Tesseract OCR), PDFs (`pypdf`), and text files for Indian tax IDs (PAN regex), Aadhaar numbers (12-digit format with **Verhoeff Dihedral Checksum** verification), payment cards (**Luhn Algorithm** validation), and bank statements.
   - **Strict Zero-Persistence Guarantee**: Raw identifiers and full OCR transcripts are never stored in the database; only redacted forms (`XXXX XXXX 1234`, `ABCDE****F`, `************1234`) are retained.
4. **Active Learning ML Junk Scoring**:
   - Starts cold with an explainable rule-based heuristic score (0.0 to 1.0).
   - Collects user Keep / Delete labels on the Suggestions page to train a personalized `RandomForestClassifier` with 5-fold cross-validation and Gini feature importance explanations.
5. **Non-Destructive Quarantine Vault**:
   - Read-only by default. Never calls `os.remove` or permanently unlinks user files. Files marked for removal are moved via `shutil.move` into `quarantine/<timestamp>/` with an audit manifest (`quarantine/manifest.json`) and can be restored in a single click.
6. **Quantitative Evaluation Benchmark**:
   - Measured against ground truth using pairwise Precision, Recall, and $F_1$-score for duplicate clustering and confusion matrices for the ML model.

---

## Architecture & Division of Labor

```
clutteriq/
  config.py                 # Paths, thresholds, and extension classifications
  models.py                 # FileRecord dataclass (Shared Data Contract)
  db.py                     # SQLite interface with WAL mode & indexes
  ssl_fix.py                # Python 3.12 Windows SSL descriptor patch
  main.py                   # Unified CLI entry point
  scanner/
    walker.py               # (A) Recursive filesystem metadata walk
  dedup/
    exact.py                # (A) Exact SHA-256 duplicate grouping
    perceptual.py           # (A) pHash perceptual near-duplicate grouping
    semantic.py             # (B) CLIP image embeddings & cosine clustering
  quality/
    blur_rank.py            # (A) Laplacian variance blur score & keeper selection
  privacy/
    sensitive.py            # (B) OCR, PyPDF, Regex, Luhn, Verhoeff, & masking
  ml/
    features.py             # (B) Leak-free tabular feature matrix construction
    junk_model.py           # (B) Heuristic scoring & Active Learning Random Forest
  safety/
    quarantine.py           # (A) Atomic quarantine moves, manifest, & restore
  bench/
    make_benchmark.py       # (A) Synthetic test folder with ground truth
    evaluate.py             # (B) Pairwise P/R/F1 benchmark & figure exporter
  app/
    app.py                  # (B) Streamlit entry shell & page router
    page_overview.py        # (B) KPI metric cards, Plotly Treemap, & histograms
    page_duplicates.py      # (A) Duplicate groups with side-by-side thumbnails
    page_privacy.py         # (B) Masked sensitive files table & quarantine action
    page_suggestions.py     # (B) Ranked recommendations & active learning feedback
    page_quarantine.py      # (A) Quarantine vault management & restore
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.11 or 3.12 (Standard laptop CPU; no GPU required).
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your machine.
  - Windows: Install via the standard installer to `C:\Program Files\Tesseract-OCR`.
  - Linux: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`

### 2. Virtual Environment & Dependencies
```bash
# Clone the repository
git clone https://github.com/your-username/Digital-Clutter-Auditor.git
cd Digital-Clutter-Auditor

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## Command-Line Usage

```bash
# 1. Scan a folder and index metadata into SQLite:
python main.py scan --root "C:\Users\YourUser\Downloads"

# 2. Run the end-to-end intelligence analysis:
python main.py analyze

# 3. Launch the interactive Streamlit dashboard:
python main.py app
# (or: streamlit run app/app.py)

# 4. Run the scientific evaluation benchmark:
python main.py evaluate
```

---

## Running Unit Tests

Run the complete test suite:

```bash
# Run all unit tests
pytest tests/

# Or run individual modules:
python tests/test_db_and_env.py          # Database & environment check
python tests/test_semantic.py            # CLIP embeddings & cosine clustering
python tests/test_sensitive.py           # Privacy auditor, Luhn & Verhoeff checksums
python tests/test_features.py            # Feature engineering & leakage checks
python tests/test_junk_model.py          # Random Forest & heuristic scoring
python tests/test_dashboard_pages.py     # Streamlit calculations & metric deduplication
python tests/test_evaluate.py            # Scientific evaluation & pairwise clustering math
```

---

## Scientific Evaluation Summary

From [`bench/BENCHMARK_REPORT.md`](bench/BENCHMARK_REPORT.md):

| Algorithm | Precision | Recall | F1-Score | Strength |
|---|---|---|---|---|
| **Exact (SHA-256)** | **1.00** | 0.35 | 0.52 | Zero false positives; catches byte-identical copies. |
| **Perceptual (pHash)** | 0.96 | 0.78 | 0.86 | Catches resizes, crops, and JPEG compression. |
| **Semantic (CLIP)** | 0.94 | 0.84 | 0.89 | Catches burst shots, angle shifts, and lighting changes. |
| **Combined (Hybrid)** | **0.95** | **0.92** | **0.93** | **Best overall $F_1$-score; unites perceptual and semantic modalities.** |

---

## Documentation & Viva Guides

- [`LEARNING_NOTES.md`](LEARNING_NOTES.md): Running glossary of machine learning and software engineering concepts with 5-question daily self-quizzes.
- [Whole-System Walkthrough & Viva Master Guide](file:///C:/Users/aksha/.gemini/antigravity/brain/803033c1-567a-4c33-8137-5c34fffe54a0/viva_and_system_walkthrough.md): Comprehensive defense document with a file journey trace, top 15 examiner questions & model answers, and a 5-minute timed live demo script.

