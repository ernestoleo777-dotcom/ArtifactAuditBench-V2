from __future__ import annotations

import csv
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
LABELS = [
    "numeric_copy_error", "relative_improvement_error", "ranking_error", "legend_swap",
    "metric_name_mismatch", "axis_scale_error", "error_bar_error", "series_data_swap",
    "aggregation_error", "significance_overclaim", "ablation_conclusion_error", "selective_reporting",
]


def safe(a, b): return a / b if b else 0.0


def image_same(a, b):
    return ImageChops.difference(Image.open(a).convert("RGB"), Image.open(b).convert("RGB")).getbbox() is None


def metrics(rows):
    tp = Counter(); fp = Counter(); fn = Counter()
    exact = clean_fp = 0
    loc_scores = []
    repairs = Counter(); repair_total = Counter()
    for row in rows:
        truth, pred = row["truth_set"], row["pred_set"]
        exact += truth == pred
        clean_fp += not truth and bool(pred)
        for label in LABELS:
            tp[label] += label in truth and label in pred
            fp[label] += label not in truth and label in pred
            fn[label] += label in truth and label not in pred
        true_items = row["truth_items"]
        pred_items = row["pred_items"]
        matched = 0
        for item in true_items:
            candidates = [p for p in pred_items if p["error_type"] == item["error_type"]]
            if candidates:
                overlap = len(set(candidates[0]["affected_elements"]) & set(item["affected_elements"]))
                union = len(set(candidates[0]["affected_elements"]) | set(item["affected_elements"]))
                matched += overlap / union if union else 1
        loc_scores.append(safe(matched, len(true_items)) if true_items else (1.0 if not pred_items else 0.0))
        for label, correct in row["repair_by_error"].items():
            repair_total[label] += 1
            repairs[label] += bool(correct)
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = safe(total_tp, total_tp + total_fp); micro_r = safe(total_tp, total_tp + total_fn)
    per_f1 = []
    for label in LABELS:
        p = safe(tp[label], tp[label] + fp[label]); r = safe(tp[label], tp[label] + fn[label])
        per_f1.append(safe(2*p*r, p+r))
    return {
        "task_count": len(rows),
        "multi_label_micro_f1": safe(2*micro_p*micro_r, micro_p+micro_r),
        "multi_label_macro_f1": sum(per_f1) / len(per_f1),
        "exact_set_match": safe(exact, len(rows)),
        "partial_localization_score": safe(sum(loc_scores), len(rows)),
        "clean_false_positive_rate": safe(clean_fp, sum(not r["truth_set"] for r in rows)),
        "per_error_repair_accuracy": {label: safe(repairs[label], repair_total[label]) for label in LABELS if repair_total[label]},
        "end_to_end_exact_repair_rate": safe(sum(r["end_to_end"] for r in rows), len(rows)),
        "average_execution_time_seconds": safe(sum(r["time"] for r in rows), len(rows)),
    }


def repair_checks(eval_dir, out_dir, truth_items):
    result = {}
    truth_types = {x["error_type"] for x in truth_items}
    if truth_types & {"numeric_copy_error", "aggregation_error", "selective_reporting"}:
        correct = (out_dir / "repaired_results.csv").exists() and (out_dir / "repaired_results.csv").read_bytes() == (eval_dir / "reference_table.csv").read_bytes()
        for label in truth_types & {"numeric_copy_error", "aggregation_error", "selective_reporting"}: result[label] = correct
    if truth_types & {"relative_improvement_error", "ranking_error", "metric_name_mismatch", "significance_overclaim", "ablation_conclusion_error"}:
        correct = (out_dir / "repaired_statement.txt").exists() and (out_dir / "repaired_statement.txt").read_bytes() == (eval_dir / "reference_statement.txt").read_bytes()
        for label in truth_types & {"relative_improvement_error", "ranking_error", "metric_name_mismatch", "significance_overclaim", "ablation_conclusion_error"}: result[label] = correct
    if truth_types & {"legend_swap", "axis_scale_error", "error_bar_error", "series_data_swap", "selective_reporting"}:
        correct = (out_dir / "repaired_visual.png").exists() and image_same(out_dir / "repaired_visual.png", eval_dir / "reference_visual.png")
        for label in truth_types & {"legend_swap", "axis_scale_error", "error_bar_error", "series_data_swap", "selective_reporting"}: result[label] = correct
    return result


def load_eval_dirs(data_root: Path, splits: list[str], manifest: Path | None) -> dict[str, list[Path]]:
    if manifest:
        data = json.loads(manifest.read_text())
        return {
            split: [data_root / "evaluator_view" / split / case_id for case_id in data["splits"].get(split, [])]
            for split in splits
        }
    return {
        split: sorted(path for path in (data_root / "evaluator_view" / split).iterdir() if path.is_dir())
        for split in splits
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data_v2"))
    parser.add_argument("--prediction-root", default=str(ROOT / "outputs/v2/rule_audit"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs/v2"))
    parser.add_argument("--splits", default="dev,test_seen,test_unseen")
    parser.add_argument("--manifest")
    parser.add_argument("--method-name", default="RuleAudit")
    parser.add_argument("--execution-failure-policy", choices=["exclude", "count_as_miss"], default="exclude")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    prediction_root = Path(args.prediction_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    manifest = Path(args.manifest).resolve() if args.manifest else None
    eval_dirs = load_eval_dirs(data_root, splits, manifest)
    rows = []
    missing = []
    for split, cases in eval_dirs.items():
        for eval_dir in cases:
            if not eval_dir.is_dir(): continue
            out_dir = prediction_root / split / eval_dir.name
            if not (out_dir / "audit_result.json").exists():
                missing.append(f"{split}/{eval_dir.name}")
                if args.execution_failure_policy == "exclude":
                    continue
            truth_items = json.loads((eval_dir / "truth.json").read_text())["errors"]
            if (out_dir / "audit_result.json").exists():
                prediction = json.loads((out_dir / "audit_result.json").read_text())
            else:
                prediction = {"predicted_errors": [], "unknown": True, "execution_time_seconds": 0.0}
            truth_set = {x["error_type"] for x in truth_items}
            pred_set = {x["error_type"] for x in prediction["predicted_errors"]}
            repairs = repair_checks(eval_dir, out_dir, truth_items)
            rows.append({
                "case_id": eval_dir.name, "split": split,
                "experiment_type": json.loads((eval_dir / "protocol.json").read_text())["experiment_type"],
                "chart_type": json.loads((eval_dir / "protocol.json").read_text())["chart_type"],
                "truth_set": truth_set, "pred_set": pred_set, "truth_items": truth_items,
                "pred_items": prediction["predicted_errors"], "repair_by_error": repairs,
                "end_to_end": truth_set == pred_set and all(repairs.values()) and not prediction["unknown"],
                "time": prediction["execution_time_seconds"], "unknown": prediction["unknown"],
            })
    overall = metrics(rows)
    by_split = {split: metrics([r for r in rows if r["split"] == split]) for split in splits}
    by_error = {label: metrics([r for r in rows if label in r["truth_set"]]) for label in LABELS}
    by_chart = {name: metrics([r for r in rows if r["chart_type"] == name]) for name in sorted({r["chart_type"] for r in rows})}
    by_experiment = {name: metrics([r for r in rows if r["experiment_type"] == name]) for name in sorted({r["experiment_type"] for r in rows})}
    single = metrics([r for r in rows if len(r["truth_set"]) == 1])
    combo = metrics([r for r in rows if len(r["truth_set"]) >= 2])
    seen_key = next((split for split in splits if "test_seen" in split), None)
    unseen_key = next((split for split in splits if "test_unseen" in split), None)
    seen = by_split[seen_key]["multi_label_micro_f1"] if seen_key else 0.0
    unseen = by_split[unseen_key]["multi_label_micro_f1"] if unseen_key else 0.0
    result = {
        "method": args.method_name,
        "overall": overall, "by_split": by_split, "single_error": single, "combination_errors": combo,
        "by_error_type": by_error, "by_chart_type": by_chart, "by_experiment_type": by_experiment,
        "unseen_template_generalization_gap": seen - unseen,
        "unseen_style_generalization_gap": seen - unseen,
        "missing_predictions": missing,
        "execution_failure_policy": args.execution_failure_policy,
        "tasks": [{**r, "truth_set": sorted(r["truth_set"]), "pred_set": sorted(r["pred_set"])} for r in rows],
    }
    (output_root / "rule_audit_evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_root / "confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["error_type", "tp", "fp", "fn"])
        for label in LABELS:
            writer.writerow([label, sum(label in r["truth_set"] and label in r["pred_set"] for r in rows),
                             sum(label not in r["truth_set"] and label in r["pred_set"] for r in rows),
                             sum(label in r["truth_set"] and label not in r["pred_set"] for r in rows)])
    failures = [r for r in rows if not r["end_to_end"]]
    with (output_root / "failure_cases.jsonl").open("w") as handle:
        for row in failures:
            serial = {**row, "truth_set": sorted(row["truth_set"]), "pred_set": sorted(row["pred_set"])}
            handle.write(json.dumps(serial, sort_keys=True) + "\n")
    print(json.dumps({"overall": overall, "by_split": by_split, "combination_errors": combo,
                      "failure_count": len(failures)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
