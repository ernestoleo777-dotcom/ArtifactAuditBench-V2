from __future__ import annotations

import csv
import itertools
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
from jsonschema import validate
from PIL import Image

from src.v2.claims import claim_context
from src.v2.core import significance
from src.v2.plots import render_plot
from src.v2.recompute import expected_table, read_csv, recompute_summary

VERSION = "2.0.0"
ALLOWED = {"protocol.json", "observations.csv", "results.csv", "visual.png", "statement.txt"}
FORBIDDEN = ("truth", "reference", "canonical", "evaluator", "clean", "corrupt", "injector", "error_type", "template")
FIGURE_ERRORS = ["legend_swap", "series_data_swap", "axis_scale_error", "error_bar_error", "selective_reporting"]


class V2AccessViolation(PermissionError):
    pass


class V2Reader:
    def __init__(self, case_dir: Path):
        self.case_dir = case_dir.resolve()
        if self.case_dir.parent.parent.name != "agent_view":
            raise V2AccessViolation("V2 RuleAudit requires a split/case under data_v2/agent_view")
        self.read_files = []

    def path(self, name: str) -> Path:
        low = name.lower()
        if name not in ALLOWED or any(token in low for token in FORBIDDEN):
            raise V2AccessViolation(name)
        path = (self.case_dir / name).resolve()
        if path.parent != self.case_dir:
            raise V2AccessViolation(name)
        if name not in self.read_files:
            self.read_files.append(name)
        return path

    def json(self, name): return json.loads(self.path(name).read_text())
    def text(self, name): return self.path(name).read_text()
    def csv(self, name): return read_csv(self.path(name))


def plot_spec(summary: dict, protocol: dict) -> dict:
    metric, chart = protocol["primary_metric"], protocol["chart_type"]
    labels = list(summary)
    values = [summary[m][metric] for m in labels]
    spec = {"chart_type": chart, "labels": labels, "values": values, "metric": metric,
            "style": protocol["render_config"], "title": f"{metric} comparison", "axis_limits": None}
    if chart == "error_bar":
        spec["errors"] = [summary[m].get("se", 0.0) for m in labels]
    if chart == "line":
        spec["series_x"] = [f"Run {i + 1}" for i in range(max(len(summary[m].get("runs", values)) for m in labels))]
        spec["series"] = [summary[m].get("runs", [summary[m][metric]] * len(spec["series_x"])) for m in labels]
    if chart == "scatter":
        spec["scatter_x"] = [list(range(len(summary[m].get("runs", [0, 1, 2])))) for m in labels]
        spec["scatter_y"] = [summary[m].get("runs", [summary[m][metric]] * 3) for m in labels]
    if chart == "confusion_heatmap":
        spec["matrix"] = summary[labels[0]].get("confusion_matrix", [[1, 0], [0, 1]])
    return spec


def apply_figure_error(spec: dict, error: str, summary: dict) -> dict:
    candidate = json.loads(json.dumps(spec))
    if error == "legend_swap" and len(candidate["labels"]) >= 2:
        candidate["labels"][0], candidate["labels"][1] = candidate["labels"][1], candidate["labels"][0]
    elif error == "series_data_swap" and len(candidate["values"]) >= 2:
        candidate["values"][0], candidate["values"][1] = candidate["values"][1], candidate["values"][0]
        if "series" in candidate:
            candidate["series"][0], candidate["series"][1] = candidate["series"][1], candidate["series"][0]
    elif error == "axis_scale_error":
        low, high = min(candidate["values"]), max(candidate["values"])
        candidate["axis_limits"] = [max(0.0, low - (high-low)*0.08), high + (high-low)*0.08]
    elif error == "error_bar_error":
        before = candidate.get("errors", [summary[m].get("se", 0) for m in candidate["labels"]])
        after = list(reversed(before))
        if after == before:
            after = [v * 1.8 + 0.002 for v in before]
        candidate["errors"] = after
    elif error == "selective_reporting":
        first = candidate["labels"][0]
        candidate["values"][0] = summary[first]["max"]
    return candidate


def pixel_equal(left: Path, right: Path) -> bool:
    a = np.asarray(Image.open(left).convert("RGB"))
    b = np.asarray(Image.open(right).convert("RGB"))
    return a.shape == b.shape and bool(np.array_equal(a, b))


def detect_figure(observed: Path, base: dict, summary: dict, work: Path) -> tuple[list[str], dict]:
    for size in range(0, 4):
        for subset in itertools.combinations(FIGURE_ERRORS, size):
            candidate = base
            valid = True
            for error in subset:
                if error == "error_bar_error" and base["chart_type"] != "error_bar":
                    valid = False
                if error == "selective_reporting" and "max" not in summary[base["labels"][0]]:
                    valid = False
                if not valid:
                    break
                candidate = apply_figure_error(candidate, error, summary)
            if not valid:
                continue
            path = work / ("candidate_" + ("clean" if not subset else "_".join(subset)) + ".png")
            render_plot(candidate, path)
            if pixel_equal(observed, path):
                return list(subset), {"matched_candidate": path.name, "repair_spec": base}
            path.unlink()
    return [], {"unknown": True, "candidate_match_count": 0}


def table_differences(observed: list[dict], expected: list[dict], summary: dict) -> list[dict]:
    expected_by = {r["model"]: r for r in expected}
    diffs = []
    for row in observed:
        exp = expected_by.get(row["model"], {})
        for column, expected_value in exp.items():
            if column == "model":
                continue
            if row.get(column) != expected_value:
                reported = float(row[column])
                kind = "numeric_copy_error"
                if column == "mean" and abs(reported - summary[row["model"]]["max"]) < 5e-7:
                    kind = "aggregation_error"
                diffs.append({"error_type": kind, "model": row["model"], "column": column,
                              "reported": reported, "expected": float(expected_value)})
    return diffs


def claim_errors(text: str, summary: dict, protocol: dict) -> list[dict]:
    metric = protocol["primary_metric"]
    context = claim_context(summary, metric)
    findings = []
    metric_mentions = set(re.findall(r"Macro-F1|Micro-F1|Weighted-F1|Accuracy|RMSE|MAE|R2", text))
    if metric not in metric_mentions and metric_mentions:
        findings.append({"error_type": "metric_name_mismatch", "affected_elements": ["metric name"],
                         "reported": sorted(metric_mentions), "expected": metric})
    models = list(summary)
    first_model = min(((text.find(model), model) for model in models if model in text), default=(-1, None))[1]
    if "trails" in text:
        trail = next((model for model in models if text.startswith(model)), None)
        inferred_best = next((model for model in models if model != trail and model in text), None)
    else:
        inferred_best = first_model
    if inferred_best and inferred_best != context["best"]:
        findings.append({"error_type": "ranking_error", "affected_elements": ["best model"],
                         "reported": inferred_best, "expected": context["best"]})
    percentages = [float(x) for x in re.findall(r"([-+]?\d+(?:\.\d+)?)%", text) if abs(float(x) - 95.0) > 1e-6]
    if percentages and all(abs(value - round(context["rel"], 2)) > 0.011 for value in percentages):
        findings.append({"error_type": "relative_improvement_error",
                         "affected_elements": ["relative_improvement_percent"],
                         "reported": percentages, "expected": round(context["rel"], 2)})
    if protocol["experiment_type"] in {"repeated_runs", "ablation"}:
        sig = significance(summary, context["best"], context["other"])
        overclaim = bool(re.search(r"statistically significant|statistically reliable difference", text, re.I))
        if overclaim and not sig:
            findings.append({"error_type": "significance_overclaim", "affected_elements": ["significance"],
                             "reported": True, "expected": False})
    if protocol["experiment_type"] == "ablation":
        if re.search(r"Module B.*largest improvement|module B.*largest improvement", text):
            findings.append({"error_type": "ablation_conclusion_error",
                             "affected_elements": ["ablation conclusion"], "reported": "Module B", "expected": "Full Model"})
    return findings


def audit_case(case_dir: Path, output_dir: Path, root: Path) -> dict:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = V2Reader(case_dir)
    protocol = reader.json("protocol.json")
    raw = reader.csv("observations.csv")
    table = reader.csv("results.csv")
    statement = reader.text("statement.txt")
    visual = reader.path("visual.png")
    summary = recompute_summary(raw, protocol)
    expected = expected_table(summary, protocol["experiment_type"])
    table_findings = table_differences(table, expected, summary)
    claim_findings = claim_errors(statement, summary, protocol)
    figure_errors, figure_evidence = detect_figure(visual, plot_spec(summary, protocol), summary, output_dir)
    predicted = []
    selective = "selective_reporting" in figure_errors
    for finding in table_findings:
        error_type = finding["error_type"]
        if error_type == "aggregation_error" and selective:
            error_type = "selective_reporting"
        predicted.append({
            "error_type": error_type, "affected_artifact": "results.csv" + ("+visual.png" if error_type == "selective_reporting" else ""),
            "affected_elements": [finding["model"], finding["column"]], "evidence": finding,
            "confidence": 1.0, "repair_available": True, "repair_result": {"value": finding["expected"]},
        })
    for finding in claim_findings:
        predicted.append({
            "error_type": finding["error_type"], "affected_artifact": "statement.txt",
            "affected_elements": finding["affected_elements"], "evidence": finding,
            "confidence": 0.95, "repair_available": True, "repair_result": {"expected": finding["expected"]},
        })
    for error in figure_errors:
        if error == "selective_reporting":
            continue
        predicted.append({
            "error_type": error, "affected_artifact": "visual.png",
            "affected_elements": {
                "legend_swap": base_elements(plot_spec(summary, protocol)),
                "series_data_swap": base_elements(plot_spec(summary, protocol)),
                "axis_scale_error": ["axis limits"],
                "error_bar_error": ["error bars"],
            }[error],
            "evidence": figure_evidence, "confidence": 1.0, "repair_available": True,
            "repair_result": {"figure": "repaired_visual.png"},
        })
    if selective and not any(p["error_type"] == "selective_reporting" for p in predicted):
        predicted.append({"error_type": "selective_reporting", "affected_artifact": "results.csv+visual.png",
                          "affected_elements": [list(summary)[0], "reported subset"], "evidence": figure_evidence,
                          "confidence": 1.0, "repair_available": True, "repair_result": {"aggregation": "mean"}})
    # Targeted deterministic repairs.
    repaired_table = [dict(row) for row in table]
    if table_findings:
        exp_by = {r["model"]: r for r in expected}
        for row in repaired_table:
            if row["model"] in exp_by:
                for finding in table_findings:
                    if finding["model"] == row["model"]:
                        row[finding["column"]] = exp_by[row["model"]][finding["column"]]
        with (output_dir / "repaired_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(repaired_table[0]))
            writer.writeheader(); writer.writerows(repaired_table)
    repaired_statement = statement
    for finding in claim_findings:
        if finding["error_type"] == "ranking_error":
            repaired_statement = repaired_statement.replace(str(finding["reported"]), str(finding["expected"]), 1)
        elif finding["error_type"] == "metric_name_mismatch":
            for wrong in finding["reported"]:
                repaired_statement = repaired_statement.replace(wrong, str(finding["expected"]))
        elif finding["error_type"] == "relative_improvement_error":
            for value in finding["reported"]:
                repaired_statement = repaired_statement.replace(f"{value:.2f}%", f"{finding['expected']:.2f}%", 1)
        elif finding["error_type"] == "significance_overclaim":
            repaired_statement = re.sub(r"The improvement is statistically significant\\.?|a statistically significant improvement",
                                        "no statistically reliable gain", repaired_statement, flags=re.I)
        elif finding["error_type"] == "ablation_conclusion_error":
            repaired_statement = repaired_statement.replace("Module B", "Full Model").replace("module B", "full model")
    if claim_findings:
        (output_dir / "repaired_statement.txt").write_text(repaired_statement)
    if figure_errors:
        render_plot(plot_spec(summary, protocol), output_dir / "repaired_visual.png")
    for path in output_dir.glob("candidate_*.png"):
        path.unlink()
    unknown = ["figure_unmatched"] if figure_evidence.get("unknown") else []
    result = {
        "case_id": protocol["case_id"], "predicted_errors": predicted, "error_detected": bool(predicted),
        "partial_success": bool(predicted) and bool(unknown), "unknown": unknown,
        "read_files": reader.read_files, "execution_status": "success",
        "execution_time_seconds": time.perf_counter() - start, "baseline_version": VERSION,
    }
    schema = json.loads((root / "configs/v2/audit_result_schema.json").read_text())
    validate(result, schema)
    write = output_dir / "audit_result.json"
    write.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def base_elements(spec: dict) -> list[str]:
    return spec["labels"][:2]
