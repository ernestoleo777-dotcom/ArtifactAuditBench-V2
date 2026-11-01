from __future__ import annotations

METRIC_DIRECTIONS = {
    "Accuracy": "maximize",
    "Precision": "maximize",
    "Recall": "maximize",
    "Macro-F1": "maximize",
    "ROC-AUC": "maximize",
    "MAE": "minimize",
    "RMSE": "minimize",
    "R2": "maximize",
}


def metric_direction(metric: str) -> str:
    try:
        return METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        raise ValueError(f"Unknown metric direction: {metric}") from exc


def rank_models(metrics: dict[str, dict[str, float]], metric: str) -> list[str]:
    reverse = metric_direction(metric) == "maximize"
    return sorted(metrics, key=lambda model: metrics[model][metric], reverse=reverse)


def comparison_values(metrics: dict[str, dict[str, float]], metric: str) -> dict:
    best, other = rank_models(metrics, metric)
    best_value = metrics[best][metric]
    other_value = metrics[other][metric]
    absolute = abs(best_value - other_value)
    relative = absolute / abs(other_value) * 100 if other_value else 0.0
    return {
        "metric": metric,
        "direction": metric_direction(metric),
        "best_model": best,
        "other_model": other,
        "best_value": best_value,
        "other_value": other_value,
        "absolute_difference": absolute,
        "percentage_point_difference": absolute * 100,
        "relative_improvement_percent": relative,
    }
