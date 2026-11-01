from __future__ import annotations

import csv
from pathlib import Path

from src.metrics import classification_metrics, regression_metrics
from src.v2.core import multiclass_metrics, run_statistics


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def model_names(rows: list[dict], experiment_type: str) -> list[str]:
    if experiment_type in {"repeated_runs", "ablation"}:
        return list(dict.fromkeys(row["model"] for row in rows))
    suffix = "__prediction"
    return [name[:-len(suffix)] for name in rows[0] if name.endswith(suffix)]


def recompute_summary(rows: list[dict[str, str]], protocol: dict) -> dict:
    experiment_type = protocol["experiment_type"]
    models = model_names(rows, experiment_type)
    if experiment_type == "binary_classification":
        y = [int(row["y_true"]) for row in rows]
        return {
            model: classification_metrics(
                y,
                [int(row[f"{model}__prediction"]) for row in rows],
                [float(row[f"{model}__probability"]) for row in rows],
            )
            for model in models
        }
    if experiment_type == "regression":
        y = [float(row["y_true"]) for row in rows]
        return {model: regression_metrics(y, [float(row[f"{model}__prediction"]) for row in rows]) for model in models}
    if experiment_type == "multiclass":
        y = [int(row["y_true"]) for row in rows]
        classes = max(y) + 1
        return {model: multiclass_metrics(y, [int(row[f"{model}__prediction"]) for row in rows], classes) for model in models}
    grouped = {model: [] for model in models}
    for row in rows:
        grouped[row["model"]].append(float(row["value"]))
    return {
        model: {"Macro-F1": run_statistics(values)["mean"], **run_statistics(values), "runs": values}
        for model, values in grouped.items()
    }


def expected_table(summary: dict, experiment_type: str) -> list[dict]:
    rows = []
    for model, values in summary.items():
        if experiment_type == "multiclass":
            row = {"model": model, **{name: f"{values[name]:.6f}" for name in ("Accuracy", "Macro-F1", "Micro-F1", "Weighted-F1")}}
            row.update({f"Recall:{name}": f"{value:.6f}" for name, value in values["per_class_recall"].items()})
        elif experiment_type in {"repeated_runs", "ablation"}:
            row = {"model": model, **{name: f"{values[name]:.6f}" for name in ("mean", "std", "se", "ci95_low", "ci95_high")}}
        else:
            row = {"model": model, **{name: f"{value:.6f}" for name, value in values.items()}}
        rows.append(row)
    return rows
