from __future__ import annotations

import math

from src.v2.recompute import expected_table

VERSION = "1.0.0"
ATOL = 5e-7


def run(reported: list[dict[str, str]], metric_facts: dict, experiment_type: str) -> dict:
    expected = expected_table(metric_facts["model_metrics"], experiment_type)
    exp_by = {row["model"]: row for row in expected}
    rep_by = {row.get("model"): row for row in reported}
    discrepancies, missing, extra = [], [], []
    for model, exp in exp_by.items():
        if model not in rep_by:
            missing.append({"model": model, "item": "row"})
            continue
        rep = rep_by[model]
        for field, value in exp.items():
            if field == "model":
                continue
            if field not in rep:
                missing.append({"model": model, "item": field})
                continue
            try:
                reported_value, expected_value = float(rep[field]), float(value)
                difference = reported_value - expected_value
                if not math.isclose(reported_value, expected_value, abs_tol=ATOL, rel_tol=1e-9):
                    discrepancies.append({
                        "model": model, "field": field, "reported": reported_value,
                        "recomputed": expected_value, "difference": difference,
                    })
            except ValueError:
                discrepancies.append({"model": model, "field": field, "reported": rep[field], "recomputed": value})
        for field in rep:
            if field not in exp:
                extra.append({"model": model, "item": field})
    for model in rep_by.keys() - exp_by.keys():
        extra.append({"model": model, "item": "row"})
    return {
        "reported_rows": reported,
        "numeric_tolerance": ATOL,
        "numeric_discrepancies": discrepancies,
        "missing_items": missing,
        "extra_items": extra,
        "metric_name_mapping": {field: field for row in reported for field in row if field != "model"},
    }
