from __future__ import annotations

from src.metrics import comparison_values


def primary_metric(task_type: str) -> str:
    return "Macro-F1" if task_type == "binary_classification" else "RMSE"


def claim_facts(metrics: dict, task_type: str) -> dict:
    return comparison_values(metrics, primary_metric(task_type))


def make_claim(metrics: dict, task_type: str) -> str:
    facts = claim_facts(metrics, task_type)
    best = facts["best_model"]
    other = facts["other_model"]
    metric = facts["metric"]
    adjective = "highest" if facts["direction"] == "maximize" else "lowest"
    if task_type == "binary_classification":
        return (
            f"{best} achieves the {adjective} {metric}. "
            f"Absolute difference: {facts['absolute_difference']:.4f}. "
            f"Percentage-point difference: {facts['percentage_point_difference']:.2f} pp. "
            f"Relative improvement: {facts['relative_improvement_percent']:.2f}% compared with {other}.\n"
        )
    return (
        f"{best} achieves the {adjective} {metric}. "
        f"Absolute reduction: {facts['absolute_difference']:.4f}. "
        f"Relative reduction: {facts['relative_improvement_percent']:.2f}% compared with {other}.\n"
    )
