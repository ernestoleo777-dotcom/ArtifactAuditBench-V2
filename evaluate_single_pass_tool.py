from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

from evaluate_direct_mllm import aggregate
from freeze_phase4b import verify

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/v2/single_pass_tool"


def rows_for(base: Path, evaluator_root: Path | None = None, splits: list[str] | None = None, selected: dict | None = None):
    evaluator_root = evaluator_root or (ROOT / "data_v2/evaluator_view")
    splits = splits or ["dev", "test_seen", "test_unseen"]
    rows = []
    for split in splits:
        cases = sorted(p for p in (evaluator_root / split).iterdir() if p.is_dir())
        if selected is not None:
            cases = [p for p in cases if p.name in selected.get(split, [])]
        for evaluator in cases:
            result_path = base / split / evaluator.name / "result.json"
            if not result_path.exists(): continue
            truth_items = json.loads((evaluator / "truth.json").read_text())["errors"]
            result = json.loads(result_path.read_text())
            pred_items = [x for x in result["predicted_errors"] if x["error_type"] not in {"clean", "unknown"}]
            protocol = json.loads((evaluator / "protocol.json").read_text())
            rows.append({
                "task_id": evaluator.name, "split": split,
                "chart_type": protocol["chart_type"], "experiment_type": protocol["experiment_type"],
                "truth": {x["error_type"] for x in truth_items},
                "prediction": {x["error_type"] for x in pred_items},
                "truth_items": truth_items, "pred_items": pred_items,
                "overall_status": result["overall_status"], "latency": result["latency_seconds"],
                "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
                "cost": result["estimated_cost"],
            })
    return rows


def incomplete_status(
    rows: list[dict],
    evaluator_root: Path | None = None,
    output_root: Path | None = None,
    splits: list[str] | None = None,
    selected: dict | None = None,
) -> dict:
    evaluator_root = evaluator_root or (ROOT / "data_v2/evaluator_view")
    output_root = output_root or OUTPUT
    splits = splits or ["dev", "test_seen", "test_unseen"]
    expected = {
        split: sorted(p.name for p in (evaluator_root / split).iterdir() if p.is_dir())
        for split in splits
    }
    if selected is not None:
        expected = {split: [task for task in tasks if task in selected.get(split, [])] for split, tasks in expected.items()}
    completed_ids = {(r["split"], r["task_id"]) for r in rows}
    missing = [
        f"{split}/{task_id}"
        for split, task_ids in expected.items()
        for task_id in task_ids
        if (split, task_id) not in completed_ids
    ]
    failed = [
        f"{split}/{task_id}"
        for split, task_ids in expected.items()
        for task_id in task_ids
        if (split, task_id) not in completed_ids
        and (output_root / split / task_id / "failure.json").exists()
    ]
    return {
        "status": "blocked_incomplete",
        "reason": "model execution results are not yet complete for all formal tasks",
        "completed_tasks": len(rows),
        "missing_tasks": len(missing),
        "missing_task_ids": missing,
        "execution_failure_tasks": len(failed),
        "execution_failure_task_ids": failed,
        "metrics": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-root", default=str(ROOT / "data_v2/evaluator_view"))
    parser.add_argument("--prediction-root", default=str(OUTPUT))
    parser.add_argument("--output-root", default=str(OUTPUT))
    parser.add_argument("--splits", default="dev,test_seen,test_unseen")
    parser.add_argument("--manifest")
    parser.add_argument("--execution-failure-policy", choices=["exclude", "count_as_miss"], default="exclude")
    args = parser.parse_args()
    prediction_root = Path(args.prediction_root).resolve()
    is_model_sweep = (ROOT / "outputs/v2/model_sweep").resolve() in prediction_root.parents
    if Path(args.evaluator_root).resolve() == (ROOT / "data_v2/evaluator_view").resolve() and not is_model_sweep:
        valid, errors = verify()
        if not valid: raise SystemExit(f"freeze violation: {errors}")
    evaluator_root = Path(args.evaluator_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    selected = json.loads(Path(args.manifest).read_text())["splits"] if args.manifest else None
    rows = rows_for(prediction_root, evaluator_root, splits, selected)
    status = incomplete_status(rows, evaluator_root, prediction_root, splits, selected)
    # Tasks that are genuinely missing (no result.json and no failure.json) still block
    # evaluation. Tasks with a confirmed, reproducible execution failure (failure.json
    # present after a cache-busted retry) are excluded from metrics rather than blocking
    # the whole evaluation, and are reported explicitly via excluded_execution_failures.
    if status["missing_tasks"] != status["execution_failure_tasks"]:
        (output_root / "evaluation.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    excluded_execution_failures = {
        "count": status["execution_failure_tasks"],
        "task_ids": status["execution_failure_task_ids"],
        "reason": "confirmed reproducible execution failure (e.g. JSON truncation at frozen output token cap), excluded from scoring",
    }
    overall = aggregate(rows)
    by_split = {s: aggregate([r for r in rows if r["split"] == s]) for s in splits}
    by_error = {}
    labels = sorted({e for r in rows for e in r["truth"]})
    for label in labels: by_error[label] = aggregate([r for r in rows if label in r["truth"]])
    by_chart = {x: aggregate([r for r in rows if r["chart_type"] == x]) for x in sorted({r["chart_type"] for r in rows})}
    by_exp = {x: aggregate([r for r in rows if r["experiment_type"] == x]) for x in sorted({r["experiment_type"] for r in rows})}
    result = {"overall": overall, "by_split": by_split, "by_error_type": by_error, "by_chart_type": by_chart, "by_experiment_type": by_exp,
              "unseen_micro_f1_gap": (
                  by_split.get("test_seen", {}).get("multi_label_micro_f1", 0.0)
                  - by_split.get("test_unseen", {}).get("multi_label_micro_f1", 0.0)
              ),
              "excluded_execution_failures": excluded_execution_failures,
              "execution_failure_policy": args.execution_failure_policy}
    (output_root / "evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_root / "per_task_results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({**r, "truth": sorted(r["truth"]), "prediction": sorted(r["prediction"])}, sort_keys=True) + "\n")
    failures = [r for r in rows if r["truth"] != r["prediction"]]
    with (output_root / "failure_cases.jsonl").open("w") as f:
        for r in failures:
            f.write(json.dumps({**r, "truth": sorted(r["truth"]), "prediction": sorted(r["prediction"])}, sort_keys=True) + "\n")
    rule = json.loads((ROOT / "outputs/v2/rule_audit_evaluation.json").read_text())["overall"]
    direct = json.loads((ROOT / "outputs/v2/direct_mllm/direct_mllm_evaluation.json").read_text())["overall"]
    with (output_root / "comparison.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Method","Micro-F1","Macro-F1","Exact Set","Localization","Clean FPR","Time","Cost"])
        w.writerow(["RuleAudit",rule["multi_label_micro_f1"],rule["multi_label_macro_f1"],rule["exact_set_match"],rule["partial_localization_score"],rule["clean_false_positive_rate"],rule["average_execution_time_seconds"],0])
        w.writerow(["Direct MLLM",direct["multi_label_micro_f1"],direct["multi_label_macro_f1"],direct["exact_set_match"],direct["partial_localization_score"],direct["clean_false_positive_rate"],direct["average_latency_seconds"],direct["total_cost"]])
        w.writerow(["Single-pass Tool Agent",overall["multi_label_micro_f1"],overall["multi_label_macro_f1"],overall["exact_set_match"],overall["partial_localization_score"],overall["clean_false_positive_rate"],overall["average_latency_seconds"],overall["total_cost"]])
    sample_path = prediction_root / "ablation_sample_manifest.json"
    sample = json.loads(sample_path.read_text())["splits"] if sample_path.exists() else {}
    ablations = {}
    for variant in ("metric", "structured"):
        variant_rows = rows_for(prediction_root / "ablations" / variant, evaluator_root, splits, sample)
        ablations[variant] = aggregate(variant_rows) if variant_rows else {"status": "not_run"}
    full_subset = [r for r in rows if r["task_id"] in sample.get(r["split"], [])]
    ablations["full"] = aggregate(full_subset)
    (output_root / "ablation_results.json").write_text(json.dumps(ablations, indent=2, sort_keys=True) + "\n")
    cost = {
        "formal_total_cost": overall["total_cost"],
        "average_input_tokens": overall["average_input_tokens"],
        "average_output_tokens": overall["average_output_tokens"],
        "average_latency_seconds": overall["average_latency_seconds"],
        "ablation_cost": sum(v.get("total_cost", 0) for k, v in ablations.items() if k != "full"),
    }
    (output_root / "cost_summary.json").write_text(json.dumps(cost, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall": overall, "by_split": by_split, "failures": len(failures), "ablations": ablations,
                       "excluded_execution_failures": excluded_execution_failures}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
