from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from eval.dataset import build_eval_set
from model.actions import RecommendedAction
from model.engine import GapRecommendationEngine


def _safe_div(top: float, bottom: float) -> float:
    return top / bottom if bottom else 0.0


def run_eval() -> Dict[str, object]:
    engine = GapRecommendationEngine("rules/rules.yaml")
    eval_set = build_eval_set()

    y_true: List[str] = []
    y_pred: List[str] = []
    drops = 0
    confidence_errors: List[float] = []

    for row in eval_set:
        pred = engine.recommend(row)
        y_true.append(str(row["label"]))
        y_pred.append(pred.recommended_action)
        if pred.hallucination_dropped:
            drops += 1
        is_correct = 1.0 if pred.recommended_action == row["label"] else 0.0
        confidence_errors.append(abs(pred.confidence - is_correct))

    classes = [action.value for action in RecommendedAction]
    cm: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for truth, pred in zip(y_true, y_pred):
        cm[truth][pred] += 1

    accuracy = _safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p), len(y_true))
    hallucination_rate = _safe_div(drops, len(y_true))
    mean_conf_calibration = sum(confidence_errors) / len(confidence_errors)

    return {
        "n": len(y_true),
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "mean_confidence_calibration": mean_conf_calibration,
        "classes": classes,
        "confusion_matrix": cm,
        "label_distribution": Counter(y_true),
    }


def write_report(metrics: Dict[str, object], report_path: str = "eval/report.md") -> None:
    classes: List[str] = metrics["classes"]  # type: ignore[assignment]
    cm = metrics["confusion_matrix"]
    lines = [
        "# Gap Classification Evaluation Report",
        "",
        f"- Samples: {metrics['n']}",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Hallucination rate: {metrics['hallucination_rate']:.3f}",
        f"- Mean confidence calibration error: {metrics['mean_confidence_calibration']:.3f}",
        "",
        "## Confusion Matrix",
        "",
        "| True \\ Pred | " + " | ".join(classes) + " |",
        "|---|" + "|".join(["---"] * len(classes)) + "|",
    ]
    for truth in classes:
        row = [str(cm[truth][pred]) for pred in classes]
        lines.append("| " + truth + " | " + " | ".join(row) + " |")

    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    metrics = run_eval()
    write_report(metrics)
    print("Evaluation complete. Report written to eval/report.md")

