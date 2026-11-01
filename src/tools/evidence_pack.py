from __future__ import annotations

import csv
import json
from pathlib import Path

from src.tools import figure_tool, metric_tool, statistics_tool, table_tool

VERSION = "1.0.0"
ALLOWED = {"protocol.json", "observations.csv", "results.csv", "statement.txt", "visual.png"}
FORBIDDEN = ("truth", "reference", "canonical", "injector", "rule_audit", "direct_mllm", "error_type", "predicted_error")


class ToolBoundaryError(PermissionError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build(case_dir: Path, root: Path) -> dict:
    case = case_dir.resolve()
    try:
        relative = case.relative_to((root / "data_v2/agent_view").resolve())
    except ValueError as exc:
        raise ToolBoundaryError("tools can only read agent_view") from exc
    if len(relative.parts) != 2 or {p.name for p in case.iterdir()} != ALLOWED:
        raise ToolBoundaryError("invalid agent case")
    protocol = json.loads((case / "protocol.json").read_text())
    observations = read_csv(case / "observations.csv")
    reported = read_csv(case / "results.csv")
    claim = (case / "statement.txt").read_text()
    metric_facts = metric_tool.run(observations, protocol)
    table_facts = table_tool.run(reported, metric_facts, protocol["experiment_type"])
    statistical_facts = statistics_tool.run(metric_facts, protocol, reported)
    figure_facts = figure_tool.run(case / "visual.png")
    uncertainties = list(figure_facts["limitations"])
    pack = {
        "task_id": protocol["case_id"],
        "metric_facts": metric_facts,
        "reported_table_facts": {
            "reported_rows": table_facts["reported_rows"],
            "missing_items": table_facts["missing_items"],
            "extra_items": table_facts["extra_items"],
            "numeric_tolerance": table_facts["numeric_tolerance"],
            "metric_name_mapping": table_facts["metric_name_mapping"],
        },
        "numeric_discrepancies": table_facts["numeric_discrepancies"],
        "ranking_facts": {
            "metric": metric_facts["primary_metric"],
            "direction": metric_facts["metric_direction"],
            "ranking": metric_facts["ranking"],
            "comparison": metric_facts["comparison"],
        },
        "aggregation_facts": {
            "required": statistical_facts["aggregation_required"],
            "model_statistics": statistical_facts["models"],
        },
        "statistical_facts": {
            "pairwise_significance": statistical_facts["pairwise_significance"],
            "selective_reporting": statistical_facts["selective_reporting"],
        },
        "figure_facts": figure_facts,
        "claim_text": claim,
        "tool_uncertainties": uncertainties,
        "read_files": ["protocol.json", "observations.csv", "results.csv", "statement.txt", "visual.png"],
        "tool_versions": {
            "metric_tool": metric_tool.VERSION, "table_tool": table_tool.VERSION,
            "statistics_tool": statistics_tool.VERSION, "figure_tool": figure_tool.VERSION,
            "evidence_pack": VERSION,
        },
    }
    text = json.dumps(pack, sort_keys=True).lower()
    if any(token in text for token in FORBIDDEN):
        raise ToolBoundaryError("answer leakage detected in evidence pack")
    return pack
