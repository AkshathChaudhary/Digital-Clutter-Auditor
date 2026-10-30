"""Command-Line Interface (CLI) entry point for ClutterIQ.

Usage:
    python main.py scan --root <folder>
    python main.py analyze
    python main.py evaluate
    python main.py app
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

# Must import ssl_fix before any networking / ML operations
import ssl_fix

from config import DB_PATH
import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clutteriq")


def cmd_scan(root_folder: str) -> None:
    """Execute filesystem metadata walk and populate SQLite database."""
    target = Path(root_folder).resolve()
    if not target.exists():
        logger.error(f"Target folder does not exist: {target}")
        sys.exit(1)

    logger.info(f"Scanning directory: {target}")
    db.init_db(DB_PATH)

    try:
        from scanner.walker import scan_folder
        count = scan_folder(str(target))
        logger.info(f"Scan finished. Indexed {count:,} files into {DB_PATH}.")
    except ImportError:
        logger.warning(
            "Person A's 'scanner/walker.py' is not yet present. "
            "Using fallback metadata scanner."
        )
        from config import kind_from_ext
        from models import FileRecord

        files_seen = 0
        for p in target.rglob("*"):
            if p.is_file() and not p.is_symlink():
                try:
                    stat = p.stat()
                    ext = p.suffix.lower().lstrip(".")
                    rec = FileRecord(
                        path=str(p.resolve()),
                        name=p.name,
                        ext=ext,
                        kind=kind_from_ext(ext),
                        size=stat.st_size,
                        created=stat.st_ctime,
                        modified=stat.st_mtime,
                        accessed=stat.st_atime,
                    )
                    db.upsert_file(rec, db_path=DB_PATH)
                    files_seen += 1
                except Exception as e:
                    logger.debug(f"Skipping {p}: {e}")
        logger.info(f"Fallback scan indexed {files_seen:,} files into {DB_PATH}.")


def cmd_analyze() -> None:
    """Run full intelligence pipeline: exact -> phash -> blur -> clip -> privacy -> ml."""
    db.init_db(DB_PATH)
    logger.info("Starting ClutterIQ End-to-End Pipeline Analysis...")

    # 1. Exact Duplicate Grouping (Person A)
    try:
        from dedup.exact import find_exact_duplicates
        logger.info("[1/6] Running Exact Deduplication (SHA-256)...")
        find_exact_duplicates()
    except ImportError:
        logger.info("[1/6] Exact Deduplication (Person A module pending). Skipping.")

    # 2. Perceptual Duplicate Grouping (Person A)
    try:
        from dedup.perceptual import find_near_duplicates
        logger.info("[2/6] Running Perceptual Deduplication (pHash)...")
        find_near_duplicates()
    except ImportError:
        logger.info("[2/6] Perceptual Deduplication (Person A module pending). Skipping.")

    # 3. Quality & Keeper Selection (Person A)
    try:
        from quality.blur_rank import choose_keepers
        logger.info("[3/6] Running Blur Variance & Keeper Selection...")
        choose_keepers()
    except ImportError:
        logger.info("[3/6] Keeper Selection (Person A module pending). Skipping.")

    # 4. Semantic Deduplication (Person B)
    logger.info("[4/6] Running Semantic Deduplication (CLIP ViT-B-32)...")
    from dedup.semantic import cluster_semantic, compute_embeddings
    compute_embeddings(db_path=DB_PATH)
    cluster_semantic(db_path=DB_PATH)

    # 5. Privacy Risk Detection (Person B)
    logger.info("[5/6] Running Privacy & Sensitive Document Auditor (OCR/PDF/Regex/Luhn)...")
    from privacy.sensitive import scan_sensitive
    scan_sensitive(db_path=DB_PATH)

    # 6. ML Junk Scoring (Person B)
    logger.info("[6/6] Computing ML Junk & Staleness Probabilities...")
    from ml.junk_model import predict_scores
    predict_scores(db_path=DB_PATH)

    logger.info("Pipeline analysis complete! Launch the dashboard with 'python main.py app'.")


def cmd_evaluate() -> None:
    """Execute scientific evaluation suite and update results.csv & figures."""
    from bench.evaluate import run_evaluation
    logger.info("Running ClutterIQ Scientific Evaluation Suite...")
    df = run_evaluation()
    print("\n" + df[["Module", "Algorithm", "Precision", "Recall", "F1-Score"]].to_string(index=False) + "\n")


def cmd_app() -> None:
    """Launch Streamlit web dashboard in browser."""
    app_path = Path(__file__).resolve().parent / "app" / "app.py"
    logger.info(f"Launching Streamlit application from {app_path}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClutterIQ: Privacy-Aware Local Digital Hygiene Auditor",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 'scan' command
    scan_parser = subparsers.add_parser("scan", help="Scan a directory and index file metadata")
    scan_parser.add_argument("--root", required=True, help="Absolute path to folder to scan")

    # 'analyze' command
    subparsers.add_parser("analyze", help="Run full deduplication, privacy, and ML scoring pipeline")

    # 'evaluate' command
    subparsers.add_parser("evaluate", help="Compute precision/recall/F1 metrics and export figures")

    # 'app' command
    subparsers.add_parser("app", help="Launch interactive Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args.root)
    elif args.command == "analyze":
        cmd_analyze()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "app":
        cmd_app()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

