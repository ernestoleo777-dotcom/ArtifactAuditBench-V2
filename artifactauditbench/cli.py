"""Offline public commands. Historical implementations remain unchanged."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_CASE = "v2_0fa6872e4f7039e6"
METHODS = ("ruleaudit", "direct", "tool", "ruleaudit_scale")


def rescore(method: str, failure_policy: str = "exclude") -> dict:
    from evaluate_direct_mllm import aggregate
    data_root = ROOT / ("data_v2_scale" if method.endswith("_scale") else "data_v2")
    manifest = json.loads((data_root / "split_manifest.json").read_text())
    expected = {(split, task) for split, tasks in manifest["splits"].items() for task in tasks}
    predictions = {}
    for line in (ROOT / "historical_predictions" / f"{method}.jsonl").read_text().splitlines():
        record = json.loads(line)
        key = (record["split"], record["task_id"])
        if key in predictions or key not in expected:
            raise ValueError(f"duplicate or unexpected prediction: {key}")
        predictions[key] = record
    rows = []
    missing = sorted(expected - predictions.keys())
    for split, task_id in sorted(expected):
        prediction = predictions.get((split, task_id))
        if prediction is None and failure_policy == "exclude":
            continue
        truth_items = json.loads((data_root / "evaluator_view" / split / task_id / "truth.json").read_text())["errors"]
        pred_items = prediction["predicted_errors"] if prediction else []
        rows.append({"truth": {e["error_type"] for e in truth_items},
                     "prediction": {e["error_type"] for e in pred_items},
                     "truth_items": truth_items, "pred_items": pred_items,
                     "force_miss": prediction is None,
                     "overall_status": prediction["overall_status"] if prediction else "unknown",
                     "latency": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0})
    metrics = aggregate(rows)
    # This release contains labels, not provider usage records. Do not publish zeros as historical costs.
    metrics = {key: value for key, value in metrics.items()
               if key in {"task_count", "multi_label_micro_f1", "multi_label_macro_f1",
                          "exact_set_match", "partial_localization_score", "clean_false_positive_rate"}}
    return {"scope": "RESCORE_SAVED_STRUCTURED_LABELS_NOT_MODEL_RERUN", "method": method,
            "planned_tasks": len(expected), "available_predictions": len(predictions),
            "failure_policy": failure_policy, "missing_task_ids": [f"{s}/{t}" for s, t in missing],
            "metrics": metrics, "provider_calls": 0}


def demo() -> dict:
    from src.v2.baseline import audit_case
    case = ROOT / "data_v2" / "agent_view" / "dev" / DEMO_CASE
    with tempfile.TemporaryDirectory(prefix="aab-v2-demo-") as temp:
        prediction = audit_case(case, Path(temp) / "prediction", ROOT)
    # Gold is read only after auditing; it is not passed to the auditor.
    truth = json.loads((ROOT / "data_v2/evaluator_view/dev" / DEMO_CASE / "truth.json").read_text())
    return {"scope": "KNOWN_DEV_CASE_DEMONSTRATION_NOT_NEW_SCIENTIFIC_RESULT",
            "task_id": DEMO_CASE, "split": "dev", "statement": (case / "statement.txt").read_text(),
            "input_files_read_by_auditor": prediction["read_files"],
            "predicted_errors": prediction["predicted_errors"],
            "reference_error_types": [e["error_type"] for e in truth["errors"]],
            "exact_error_set_match": {e["error_type"] for e in prediction["predicted_errors"]}
                                     == {e["error_type"] for e in truth["errors"]},
            "provider_calls": 0}


def run_script(script: str, args: list[str]) -> int:
    return subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m artifactauditbench", description="ArtifactAuditBench V2 offline tools; no provider calls.")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("demo", help="Audit one known dev example / 离线演示")
    d.add_argument("--json", action="store_true")
    v = commands.add_parser("validate", help="Validate frozen data / 校验数据")
    v.add_argument("--scale", action="store_true")
    r = commands.add_parser("ruleaudit", help="Run RuleAudit into a fresh runs/ directory")
    r.add_argument("--split", choices=("dev", "test_seen", "test_unseen", "all"), default="dev")
    r.add_argument("--output", required=True, type=Path)
    s = commands.add_parser("rescore", help="Score saved labels, without rerunning models")
    s.add_argument("--method", choices=METHODS, required=True)
    s.add_argument("--failure-policy", choices=("exclude", "count_as_miss"), default="exclude")
    args = parser.parse_args(argv)
    if args.command == "validate":
        return run_script("validate_v2.py", ["--data-root", str(ROOT / "data_v2_scale"), "--expected-count", "360"] if args.scale else ["--strict-v2-120"])
    if args.command == "ruleaudit":
        output = args.output.resolve()
        runs_root = (ROOT / "runs").resolve()
        if not output.is_relative_to(runs_root) or output == runs_root:
            parser.error("--output must be a fresh subdirectory of this checkout's runs/ directory")
        if output.exists():
            parser.error("output already exists; choose a new runs/ subdirectory")
        splits = "dev,test_seen,test_unseen" if args.split == "all" else args.split
        result = run_script("run_rule_audit_v2.py", ["--splits", splits, "--output-root", str(output)])
        if result:
            return result
        return run_script("evaluate_v2.py", ["--splits", splits, "--prediction-root", str(output), "--output-root", str(output / "evaluation")])
    result = demo() if args.command == "demo" else rescore(args.method, args.failure_policy)
    if args.command == "demo" and not args.json:
        print("ArtifactAuditBench V2 / 离线演示（已知开发集案例，非新增科学结果）")
        print(f"Task: {result['task_id']}")
        print(result["statement"])
        print("Input files / 审计输入:", ", ".join(result["input_files_read_by_auditor"]))
        print("Predicted / 预测:", ", ".join(e["error_type"] for e in result["predicted_errors"]) or "none")
        print("Reference / 参考:", ", ".join(result["reference_error_types"]))
        print("Exact error-set match / 错误集合一致:", result["exact_error_set_match"])
        print("Provider calls / 模型调用: 0")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
