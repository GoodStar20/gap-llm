from __future__ import annotations

from eval.run_eval import run_eval


def test_eval_runs_minimum_size() -> None:
    metrics = run_eval()
    assert metrics["n"] >= 50
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["hallucination_rate"] <= 1.0
    assert 0.0 <= metrics["mean_confidence_calibration"] <= 1.0

