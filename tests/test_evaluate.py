"""Unit test for benchmarking and evaluation suite (bench/evaluate.py).

Verifies:
1. Mathematical precision of pairwise clustering evaluation (compute_pairwise_metrics).
2. End-to-end execution of run_evaluation().
3. Generation of results.csv, HTML interactive figures, and BENCHMARK_REPORT.md.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ssl_fix

from bench.evaluate import (
    BENCH_DIR,
    FIGURES_DIR,
    REPORT_MD,
    RESULTS_CSV,
    compute_pairwise_metrics,
    run_evaluation,
)


def test_pairwise_metrics():
    print("Testing pairwise clustering evaluation math...")

    # True duplicate clusters: {A, B} in group 1, {C, D} in group 2
    # Total pairs among 4 items = 4 * 3 / 2 = 6 pairs:
    # (A, B) -> True duplicate
    # (C, D) -> True duplicate
    # (A, C), (A, D), (B, C), (B, D) -> True non-duplicates
    true_labels = {"A": 1, "B": 1, "C": 2, "D": 2}

    # Case 1: Perfect Prediction
    pred_perfect = {"A": 10, "B": 10, "C": 20, "D": 20}
    res_perf = compute_pairwise_metrics(true_labels, pred_perfect)
    assert res_perf["precision"] == 1.0
    assert res_perf["recall"] == 1.0
    assert res_perf["f1"] == 1.0
    assert res_perf["tp"] == 2
    assert res_perf["fp"] == 0
    assert res_perf["fn"] == 0

    # Case 2: Missed one pair (C and D not grouped)
    pred_missed = {"A": 10, "B": 10, "C": 20, "D": 21}
    res_miss = compute_pairwise_metrics(true_labels, pred_missed)
    assert res_miss["precision"] == 1.0  # Only predicted (A, B) which is true
    assert res_miss["recall"] == 0.5  # Found 1 of 2 true pairs
    assert round(res_miss["f1"], 2) == 0.67

    # Case 3: False Positive (grouped A, B, and C together)
    pred_fp = {"A": 10, "B": 10, "C": 10, "D": 20}
    # Predicted pairs: (A,B), (A,C), (B,C) -> 1 TP, 2 FP. Missed (C,D) -> 1 FN.
    res_fp = compute_pairwise_metrics(true_labels, pred_fp)
    assert round(res_fp["precision"], 2) == 0.33  # 1 / (1 + 2)
    assert round(res_fp["recall"], 2) == 0.50  # 1 / (1 + 1)

    print(" Pairwise clustering math tests passed!")


def test_end_to_end_evaluation_run():
    print("\nTesting run_evaluation() and artifact generation...")
    df = run_evaluation()

    assert not df.empty, "Evaluation table is empty!"
    assert RESULTS_CSV.exists(), f"Missing {RESULTS_CSV}"
    assert REPORT_MD.exists(), f"Missing {REPORT_MD}"
    assert (FIGURES_DIR / "duplicate_comparison.html").exists(), "Missing duplicate comparison chart!"
    assert (FIGURES_DIR / "runtime_benchmark.html").exists(), "Missing runtime benchmark chart!"
    assert (FIGURES_DIR / "confusion_matrix.html").exists(), "Missing confusion matrix chart!"

    # Check contents of report
    report_text = REPORT_MD.read_text(encoding="utf-8")
    assert "Exact (SHA-256)" in report_text
    assert "Perceptual (pHash)" in report_text
    assert "Semantic (CLIP)" in report_text
    assert "Privacy Auditor Evaluation" in report_text

    print(" End-to-end evaluation run passed successfully!")


if __name__ == "__main__":
    test_pairwise_metrics()
    test_end_to_end_evaluation_run()

