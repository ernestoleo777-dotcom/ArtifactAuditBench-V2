from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _division(num: float, den: float) -> float:
    return num / den if den else 0.0


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if not len(positives) or not len(negatives):
        return 0.5
    comparisons = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], probabilities: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(y_pred), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    accuracy = float(np.mean(y == p))
    per_class = []
    recalls = []
    for label in (0, 1):
        tp = int(((y == label) & (p == label)).sum())
        fp = int(((y != label) & (p == label)).sum())
        fn = int(((y == label) & (p != label)).sum())
        precision = _division(tp, tp + fp)
        recall = _division(tp, tp + fn)
        f1 = _division(2 * precision * recall, precision + recall)
        per_class.append((precision, f1))
        recalls.append(recall)
    return {
        "Accuracy": accuracy,
        "Precision": per_class[1][0],
        "Recall": recalls[1],
        "Macro-F1": float(np.mean([item[1] for item in per_class])),
        "ROC-AUC": _auc(y, scores),
    }


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    errors = y - p
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(math.sqrt(np.mean(errors**2))),
        "R2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
    }


def compute_metrics(rows: list[dict], task_type: str) -> dict[str, dict[str, float]]:
    y = [row["y_true"] for row in rows]
    if task_type == "binary_classification":
        return {
            "Model A": classification_metrics(y, [r["model_a_prediction"] for r in rows], [r["model_a_probability"] for r in rows]),
            "Model B": classification_metrics(y, [r["model_b_prediction"] for r in rows], [r["model_b_probability"] for r in rows]),
        }
    return {
        "Model A": regression_metrics(y, [r["model_a_prediction"] for r in rows]),
        "Model B": regression_metrics(y, [r["model_b_prediction"] for r in rows]),
    }
