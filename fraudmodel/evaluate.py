"""Evaluation metrics for an imbalanced fraud problem.

Accuracy is useless here: a model that says "never fraud" is 98% accurate on 2% fraud.
We look at precision (how many alerts are real), recall (how much fraud we catch),
the false positive rate (genuine customers we annoy) and PR-AUC (ranking quality).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

from .config import QUALITY_BAR


def threshold_for_precision(y_true: np.ndarray, scores: np.ndarray, target_precision: float) -> float:
    """Lowest score threshold that still reaches the target precision (so recall is as high as possible)."""
    precision, _recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(precision[:-1] >= target_precision)[0]
    if len(ok) == 0:
        return float(thresholds[-1])
    return float(thresholds[ok[0]])


def evaluate(y_true: np.ndarray, scores: np.ndarray, review_threshold: float, block_threshold: float) -> dict:
    flagged = scores >= review_threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    blocked = scores >= block_threshold
    b_tn, b_fp, b_fn, b_tp = confusion_matrix(y_true, blocked, labels=[0, 1]).ravel()

    def ratio(a: float, b: float) -> float:
        return round(float(a / b), 4) if b else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "false_positive_rate": ratio(fp, fp + tn),
        "block_precision": ratio(b_tp, b_tp + b_fp),
        "block_recall": ratio(b_tp, b_tp + b_fn),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def meets_quality_bar(metrics: dict, bar: dict = QUALITY_BAR) -> bool:
    return (
        metrics["pr_auc"] >= bar["min_pr_auc"]
        and metrics["precision"] >= bar["min_precision"]
        and metrics["recall"] >= bar["min_recall"]
        and metrics["false_positive_rate"] <= bar["max_fpr"]
    )
