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
    """Scan a directory and store file metadata in SQLite."""

    target = Path(root_folder).expanduser()

    logger.info(f"Scanning directory: {target}")

    try:
        from scanner.walker import scan_directory

        # 1. Initialize the database.
        db.init_db(DB_PATH)

        # 2. Discover files and create FileRecord objects.
        records, errors = scan_directory(target)

        # 3. Save every record into SQLite.
        saved_count = 0

        for record in records:
            db.upsert_file(record, db_path=DB_PATH)
            saved_count += 1

        # 4. Report the results.
        logger.info(
            f"Scan completed. Indexed {saved_count:,} files "
            f"into {DB_PATH}."
        )

        logger.info(f"Files discovered: {len(records)}")
        logger.info(f"Filesystem errors: {len(errors)}")

        for error in errors:
            logger.warning(
                f"Could not process {error['path']}: {error['error']}"
            )

    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.error(f"Invalid scan directory: {exc}")
        sys.exit(1)



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

