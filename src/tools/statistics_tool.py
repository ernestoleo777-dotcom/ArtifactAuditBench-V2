from __future__ import annotations

from src.v2.core import significance

VERSION = "1.0.0"


def run(metric_facts: dict, protocol: dict, reported_rows: list[dict]) -> dict:
    summary = metric_facts["model_metrics"]
    result = {
        "aggregation_required": "mean" if protocol["experiment_type"] in {"repeated_runs", "ablation"} else "not_applicable",
        "models": {},
        "pairwise_significance": [],
        "selective_reporting": {"expected_run_coverage": None, "reported_run_coverage": None},
    }
    for model, values in summary.items():
        if "runs" in values:
            result["models"][model] = {
                "run_count": len(values["runs"]), "runs": values["runs"],
                "mean": values["mean"], "max": values["max"], "median": values["median"],
                "std": values["std"], "se": values["se"],
                "ci95": [values["ci95_low"], values["ci95_high"]],
            }
    models = list(result["models"])
    for i, first in enumerate(models):
        for second in models[i + 1:]:
            result["pairwise_significance"].append({
                "models": [first, second],
                "ci95_nonoverlap": significance(summary, first, second),
            })
    if models:
        result["selective_reporting"] = {
            "expected_run_coverage": {model: len(summary[model]["runs"]) for model in models},
            "reported_aggregate_rows": len(reported_rows),
            "raw_runs_available": True,
        }
    return result
