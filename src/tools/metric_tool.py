from __future__ import annotations

from src.v2.claims import claim_context
from src.v2.core import METRIC_DIRECTIONS, rank_models
from src.v2.recompute import recompute_summary

VERSION = "1.0.0"


def run(rows: list[dict[str, str]], protocol: dict) -> dict:
    summary = recompute_summary(rows, protocol)
    metric = protocol["primary_metric"]
    context = claim_context(summary, metric)
    return {
        "primary_metric": metric,
        "metric_direction": METRIC_DIRECTIONS[metric],
        "model_metrics": summary,
        "ranking": rank_models(summary, metric),
        "comparison": {
            "best_model": context["best"], "second_model": context["other"],
            "best_value": context["bestv"], "second_value": context["otherv"],
            "absolute_difference": context["abs"],
            "percentage_point_difference": context["pp"],
            "relative_improvement_percent": context["rel"],
        },
    }
