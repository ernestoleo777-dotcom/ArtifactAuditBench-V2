from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

ERROR_TYPES = [
    "numeric_copy_error", "relative_improvement_error", "ranking_error", "legend_swap",
    "metric_name_mismatch", "axis_scale_error", "error_bar_error", "series_data_swap",
    "aggregation_error", "significance_overclaim", "ablation_conclusion_error", "selective_reporting",
]

METRIC_DIRECTIONS = {
    "Accuracy": "maximize", "Precision": "maximize", "Recall": "maximize",
    "Macro-F1": "maximize", "Micro-F1": "maximize", "Weighted-F1": "maximize",
    "ROC-AUC": "maximize", "R2": "maximize", "MAE": "minimize", "RMSE": "minimize",
}

NEUTRAL_NAMES = [
    "Method P", "System K", "Variant R", "Approach T", "Model Q",
    "Procedure N", "System V", "Method X", "Variant C", "Approach H",
]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def multiclass_metrics(y_true, y_pred, classes: int) -> dict:
    y = np.asarray(y_true, int)
    p = np.asarray(y_pred, int)
    recalls, f1s, supports = [], [], []
    matrix = np.zeros((classes, classes), dtype=int)
    for truth, pred in zip(y, p):
        matrix[truth, pred] += 1
    for label in range(classes):
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(int((y == label).sum()))
    accuracy = float(np.mean(y == p))
    return {
        "Accuracy": accuracy,
        "Macro-F1": float(np.mean(f1s)),
        "Micro-F1": accuracy,
        "Weighted-F1": float(np.average(f1s, weights=supports)),
        "per_class_recall": {f"Class {i}": float(recalls[i]) for i in range(classes)},
        "confusion_matrix": matrix.tolist(),
    }


def run_statistics(values: list[float]) -> dict:
    array = np.asarray(values, float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    se = std / math.sqrt(len(array)) if len(array) else 0.0
    margin = NormalDist().inv_cdf(0.975) * se
    return {
        "mean": mean, "std": std, "se": se,
        "ci95_low": mean - margin, "ci95_high": mean + margin,
        "median": float(np.median(array)), "max": float(np.max(array)),
    }


def rank_models(summary: dict, metric: str) -> list[str]:
    reverse = METRIC_DIRECTIONS.get(metric, "maximize") == "maximize"
    return sorted(summary, key=lambda model: summary[model][metric], reverse=reverse)


def significance(summary: dict, first: str, second: str) -> bool:
    a, b = summary[first], summary[second]
    if not all(key in a and key in b for key in ("ci95_low", "ci95_high")):
        return False
    return a["ci95_low"] > b["ci95_high"] or b["ci95_low"] > a["ci95_high"]
