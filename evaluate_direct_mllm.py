from __future__ import annotations

import csv
import json
import argparse
from collections import Counter
from pathlib import Path

from freeze_v2_benchmark import verify_manifest
from src.v2.core import ERROR_TYPES

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/v2/direct_mllm"


def safe(a, b):
    return a / b if b else 0.0


def aggregate(rows: list[dict]) -> dict:
    labels = ERROR_TYPES
    tp = Counter(); fp = Counter(); fn = Counter()
    exact = clean_fp = unknown = 0
    localization = []
    for row in rows:
        truth, pred = row["truth"], row["prediction"]
        force_miss = row.get("force_miss", False)
        exact += truth == pred and not force_miss
        clean_fp += not truth and bool(pred)
        unknown += row["overall_status"] == "unknown"
        for label in labels:
            tp[label] += label in truth and label in pred
            fp[label] += label not in truth and label in pred
            fn[label] += label in truth and label not in pred
        score = 0.0
        if force_miss:
            score = 0.0
        elif not row["truth_items"]:
            score = 1.0 if not row["pred_items"] else 0.0
        else:
            for true_item in row["truth_items"]:
                matches = [p for p in row["pred_items"] if p["error_type"] == true_item["error_type"]]
                if matches:
                    left, right = set(true_item["affected_elements"]), set(matches[0]["affected_elements"])
                    score += safe(len(left & right), len(left | right))
            score /= len(row["truth_items"])
        localization.append(score)
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p = safe(total_tp, total_tp + total_fp); r = safe(total_tp, total_tp + total_fn)
    macro = []
    for label in labels:
        lp = safe(tp[label], tp[label] + fp[label]); lr = safe(tp[label], tp[label] + fn[label])
        macro.append(safe(2 * lp * lr, lp + lr))
    return {
        "task_count": len(rows),
        "multi_label_micro_f1": safe(2 * p * r, p + r),
        "multi_label_macro_f1": safe(sum(macro), len(macro)),
        "exact_set_match": safe(exact, len(rows)),
        "partial_localization_score": safe(sum(localization), len(rows)),
        "clean_false_positive_rate": safe(clean_fp, sum(not row["truth"] for row in rows)),
        "unknown_count": unknown,
        "average_latency_seconds": safe(sum(row["latency"] for row in rows), len(rows)),
        "average_input_tokens": safe(sum(row["input_tokens"] for row in rows), len(rows)),
        "average_output_tokens": safe(sum(row["output_tokens"] for row in rows), len(rows)),
        "total_cost": sum(row["cost"] for row in rows),
        "per_error": {
            label: {"tp": tp[label], "fp": fp[label], "fn": fn[label]}
            for label in labels
        },
    }


def load_expected(evaluator_root: Path, splits: list[str], manifest_path: Path | None) -> dict[str, list[Path]]:
    if manifest_path:
        manifest = json.loads(manifest_path.read_text())
        return {
            split: [evaluator_root / split / task_id for task_id in manifest["splits"].get(split, [])]
            for split in splits
        }
    return {split: sorted(path for path in (evaluator_root / split).iterdir() if path.is_dir()) for split in splits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-root", default=str(ROOT / "data_v2/evaluator_view"))
    parser.add_argument("--prediction-root", default=str(OUTPUT))
    parser.add_argument("--output-root", default=str(OUTPUT))
    parser.add_argument("--splits", default="dev,test_seen,test_unseen")
    parser.add_argument("--manifest")
    parser.add_argument("--execution-failure-policy", choices=["exclude", "count_as_miss"], default="exclude")
    args = parser.parse_args()
    if Path(args.evaluator_root).resolve() == (ROOT / "data_v2/evaluator_view").resolve():
        valid, errors = verify_manifest()
        if not valid:
            raise SystemExit(f"benchmark freeze violation: {errors}")
    evaluator_root = Path(args.evaluator_root).resolve()
    prediction_root = Path(args.prediction_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    expected = load_expected(evaluator_root, splits, Path(args.manifest).resolve() if args.manifest else None)
    missing = [
        f"{split}/{case.name}"
        for split, cases in expected.items()
        for case in cases
        if not (prediction_root / split / case.name / "result.json").exists()
    ]
    execution_failures = [
        task
        for task in missing
        if (prediction_root / task / "failure.json").exists()
    ]
    if missing:
        if args.execution_failure_policy == "exclude":
            evaluable = {
                split: [case for case in cases if (prediction_root / split / case.name / "result.json").exists()]
                for split, cases in expected.items()
            }
            if any(evaluable.values()):
                expected = evaluable
            else:
                evaluable = None
        else:
            unconfirmed_missing = sorted(set(missing) - set(execution_failures))
            evaluable = {} if not unconfirmed_missing else None
        if evaluable is None:
            result = {
                "status": "blocked_incomplete",
                "reason": "real model results are unavailable",
                "completed_tasks": sum(len(cases) for cases in expected.values()) - len(missing),
                "missing_tasks": len(missing),
                "missing_task_ids": missing,
                "execution_failure_tasks": len(execution_failures),
                "execution_failure_task_ids": execution_failures,
                "metrics": None,
            }
            (output_root / "direct_mllm_evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            (output_root / "per_task_results.jsonl").write_text("")
            (output_root / "failure_cases.jsonl").write_text("")
            (output_root / "cost_summary.json").write_text(json.dumps({
                "status": "no_api_runs", "total_cost": 0.0, "completed_tasks": 0,
            }, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result, indent=2))
            return
    rows = []
    for split, cases in expected.items():
        for evaluator in cases:
            truth_items = json.loads((evaluator / "truth.json").read_text())["errors"]
            result_path = prediction_root / split / evaluator.name / "result.json"
            failure_path = prediction_root / split / evaluator.name / "failure.json"
            if result_path.exists():
                prediction = json.loads(result_path.read_text())
                pred_items = [item for item in prediction["predicted_errors"] if item["error_type"] not in {"clean", "unknown"}]
                status = prediction["overall_status"]
                latency = prediction["latency_seconds"]
                input_tokens = prediction["input_tokens"]
                output_tokens = prediction["output_tokens"]
                cost = prediction["estimated_cost"]
                force_miss = False
                execution_status = prediction.get("execution_status", "success")
                failure_stage = None
            else:
                failure = json.loads(failure_path.read_text())
                pred_items = []
                status = "execution_failed"
                latency = failure.get("latency_seconds", 0.0)
                input_tokens = failure.get("input_tokens", 0)
                output_tokens = failure.get("output_tokens", 0)
                cost = failure.get("estimated_cost", 0.0)
                force_miss = True
                execution_status = "failed"
                failure_stage = failure.get("failure_stage")
            rows.append({
                "task_id": evaluator.name, "split": split,
                "chart_type": json.loads((evaluator / "protocol.json").read_text())["chart_type"],
                "experiment_type": json.loads((evaluator / "protocol.json").read_text())["experiment_type"],
                "truth": {item["error_type"] for item in truth_items},
                "prediction": {item["error_type"] for item in pred_items},
                "truth_items": truth_items, "pred_items": pred_items,
                "overall_status": status,
                "latency": latency, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "cost": cost,
                "force_miss": force_miss,
                "execution_status": execution_status,
                "failure_stage": failure_stage,
            })
    overall = aggregate(rows)
    by_split = {split: aggregate([row for row in rows if row["split"] == split]) for split in expected}
    by_chart = {name: aggregate([row for row in rows if row["chart_type"] == name]) for name in sorted({r["chart_type"] for r in rows})}
    by_experiment = {name: aggregate([row for row in rows if row["experiment_type"] == name]) for name in sorted({r["experiment_type"] for r in rows})}
    result = {"status": "complete", "overall": overall, "by_split": by_split, "by_chart_type": by_chart, "by_experiment_type": by_experiment}
    result["missing_predictions"] = missing
    result["execution_failures"] = {
        "count": len(execution_failures),
        "task_ids": execution_failures,
        "policy": args.execution_failure_policy,
    }
    result["execution_failure_policy"] = args.execution_failure_policy
    (output_root / "direct_mllm_evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_root / "per_task_results.jsonl").open("w") as handle:
        for row in rows:
            serial = {**row, "truth": sorted(row["truth"]), "prediction": sorted(row["prediction"])}
            handle.write(json.dumps(serial, sort_keys=True) + "\n")
    failures = [row for row in rows if row["force_miss"] or row["truth"] != row["prediction"]]
    with (output_root / "failure_cases.jsonl").open("w") as handle:
        for row in failures:
            serial = {**row, "truth": sorted(row["truth"]), "prediction": sorted(row["prediction"])}
            handle.write(json.dumps(serial, sort_keys=True) + "\n")
    rule = json.loads((ROOT / "outputs/v2/rule_audit_evaluation.json").read_text())["overall"]
    with (output_root / "rule_vs_mllm.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", "Micro-F1", "Macro-F1", "Exact Set", "Clean FPR", "Localization", "Time", "Cost"])
        writer.writerow(["RuleAudit", rule["multi_label_micro_f1"], rule["multi_label_macro_f1"],
                         rule["exact_set_match"], rule["clean_false_positive_rate"],
                         rule["partial_localization_score"], rule["average_execution_time_seconds"], 0.0])
        writer.writerow(["Direct MLLM", overall["multi_label_micro_f1"], overall["multi_label_macro_f1"],
                         overall["exact_set_match"], overall["clean_false_positive_rate"],
                         overall["partial_localization_score"], overall["average_latency_seconds"], overall["total_cost"]])
    (output_root / "cost_summary.json").write_text(json.dumps({
        "status": "complete", "total_cost": overall["total_cost"],
        "average_input_tokens": overall["average_input_tokens"],
        "average_output_tokens": overall["average_output_tokens"],
        "average_latency_seconds": overall["average_latency_seconds"],
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall": overall, "by_split": by_split}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
