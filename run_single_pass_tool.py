from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from freeze_phase4b import verify
from freeze_single_pass_prompt import verify_prompt
from src.agents.single_pass_tool import SinglePassToolRunner
from src.providers import OpenRouterProvider
from utils.finalize import append_execution_failure, run_finalize

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/v2/single_pass_tool"
STATE = OUTPUT / "run_state.json"


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_").replace(" ", "_")


def state(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_manifest(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


def split_tasks(split: str, manifest: dict | None) -> list[str]:
    if manifest:
        return list(manifest["splits"].get(split, []))
    return sorted(p.name for p in (ROOT / "data_v2/agent_view" / split).iterdir() if p.is_dir())


def manifest_tasks(manifest: dict | None) -> list[tuple[str, str]]:
    if not manifest:
        return [
            (split, task_id)
            for split in ("dev", "test_seen", "test_unseen")
            for task_id in split_tasks(split, None)
        ]
    return [
        (split, task_id)
        for split, task_ids in manifest["splits"].items()
        for task_id in task_ids
    ]


def is_test_seen_completed(st: dict) -> bool:
    """Read test_seen completion out of run_state.json, tolerant of multiple shapes.

    Accepts (any one is sufficient):
      a) {"test_seen": "completed"}
      b) {"test_seen_completed": true}
      c) {"test_seen": {"status": "completed"}}
      d) {"test_seen_complete": true}  -- the boolean key this runner itself writes
    """
    test_seen_value = st.get("test_seen")
    if test_seen_value == "completed":
        return True
    if isinstance(test_seen_value, dict) and test_seen_value.get("status") == "completed":
        return True
    if st.get("test_seen_completed") is True:
        return True
    if st.get("test_seen_complete") is True:
        return True
    return False


def run_tasks(runner, split, tasks, output_root, variant):
    results = []
    for task_id in tasks:
        try:
            results.append(runner.run(split, task_id, output_root / split / task_id, variant))
        except Exception as exc:
            # KeyboardInterrupt/SystemExit are not Exception subclasses and propagate untouched.
            append_execution_failure(
                output_root / "execution_failures.jsonl",
                task_id=task_id,
                stage=split,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            results.append({
                "task_id": task_id,
                "execution_status": "failed",
                "failure_stage": "unhandled_runtime_error",
                "error_type": type(exc).__name__,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0,
            })
    return results


def run_task_pairs(runner, task_pairs, output_root, variant):
    results = []
    for split, task_id in task_pairs:
        try:
            results.append(runner.run(split, task_id, output_root / split / task_id, variant))
        except Exception as exc:
            append_execution_failure(
                output_root / "execution_failures.jsonl",
                task_id=task_id,
                stage=split,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            results.append({
                "task_id": task_id,
                "execution_status": "failed",
                "failure_stage": "unhandled_runtime_error",
                "error_type": type(exc).__name__,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0,
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["smoke", "dev-smoke", "dev", "test_seen", "test_unseen", "ablation"])
    parser.add_argument("--variant", default="full", choices=["metric", "structured", "full"])
    parser.add_argument("--manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--model")
    parser.add_argument("--evidence-root", default=str(ROOT / "outputs/v2/single_pass_tool/evidence_packs"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/models/single_pass_tool.json").read_text())
    if args.model:
        os.environ[cfg["model_env"]] = args.model
    configured_model = os.environ.get(cfg["model_env"])
    output = Path(args.output_root).resolve() if args.output_root else (
        ROOT / "outputs/v2/model_sweep" / model_slug(configured_model) / "single_pass_tool"
        if args.model and configured_model else OUTPUT
    )
    is_model_sweep = (ROOT / "outputs/v2/model_sweep").resolve() in output.parents
    if not is_model_sweep:
        valid, errors = verify()
        if not valid: raise SystemExit(f"freeze violation: {errors}")
    manifest = load_manifest(args.manifest)
    evidence_root = Path(args.evidence_root).resolve()
    state_path = output / "run_state.json"
    st = state(state_path)
    if args.list:
        if args.stage == "smoke":
            task_pairs = manifest_tasks(manifest)
        else:
            split = "dev" if args.stage in {"dev-smoke", "dev"} else args.stage
            task_pairs = [(split, task_id) for task_id in split_tasks(split, manifest)]
            if args.stage == "dev-smoke":
                task_pairs = task_pairs[:3]
        print(json.dumps({
            "status": "list",
            "stage": args.stage,
            "output_root": str(output),
            "evidence_root": str(evidence_root),
            "model": configured_model or "<ARTIFACT_AUDIT_MODEL unset>",
            "tasks": [{"split": split, "task_id": task_id} for split, task_id in task_pairs],
        }, indent=2, sort_keys=True))
        return
    provider = OpenRouterProvider(cfg)
    runner = SinglePassToolRunner(
        ROOT,
        provider,
        cache_dir=output / "cache",
        evidence_root=evidence_root,
        enforce_phase4b_freeze=not is_model_sweep,
    )
    if args.stage == "dev" and not st.get("dev_smoke_complete"): raise SystemExit("dev-smoke required")
    if args.stage in {"test_seen", "test_unseen", "ablation"}:
        ok, errs = verify_prompt(True)
        if not ok or not st.get("dev_complete"): raise SystemExit(f"frozen dev prompt required: {errs}")
    if args.stage == "test_unseen" and not is_test_seen_completed(st): raise SystemExit("test_seen required")
    if args.stage in {"test_seen", "test_unseen"} and st.get(f"{args.stage}_complete"): raise SystemExit(f"{args.stage} already complete")
    tasks = []
    if args.stage == "smoke":
        task_pairs = manifest_tasks(manifest)
        results = run_task_pairs(runner, task_pairs, output, args.variant)
        successes = [result for result in results if result.get("execution_status") == "success"]
        failures = [result for result in results if result.get("execution_status") == "failed"]
        st["smoke_complete"] = len(successes) == len(task_pairs)
        st["smoke_task_count"] = len(task_pairs)
        st["smoke_success_count"] = len(successes)
        st["smoke_failure_count"] = len(failures)
        save(st, state_path)
        final_manifests = []
        for pair_split in sorted({split for split, _ in task_pairs}):
            final_manifests.append(run_finalize(
                pair_split,
                output / pair_split,
                jsonl_path=output / "per_task_results.jsonl",
                csv_path=output / "results.csv",
            ))
        print(json.dumps({
            "stage": args.stage,
            "tasks": len(results),
            "successful_tasks": len(successes),
            "failed_tasks": len(failures),
            "tokens": sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in results),
            "cost": sum(r.get("estimated_cost", 0.0) for r in results),
            "model": provider.model,
            "final_manifests": final_manifests,
        }, indent=2))
        return
    if args.stage in {"dev-smoke", "dev"}:
        split = "dev"; tasks = split_tasks(split, manifest)
        if args.stage == "dev-smoke": tasks = tasks[:3]
    elif args.stage in {"test_seen", "test_unseen"}:
        split = args.stage; tasks = split_tasks(split, manifest)
    else:
        sample = json.loads((output / "ablation_sample_manifest.json").read_text())
        totals = []
        for variant in ("metric", "structured"):
            for split, ids in sample["splits"].items():
                for task_id in ids:
                    result = runner.run(split, task_id, output / "ablations" / variant / split / task_id, variant)
                    totals.append(result)
        print(json.dumps({"stage": "ablation", "tasks": len(totals), "cost": sum(r["estimated_cost"] for r in totals)}, indent=2)); return
    results = run_tasks(runner, split, tasks, output, args.variant)
    successes = [result for result in results if result.get("execution_status") == "success"]
    failures = [result for result in results if result.get("execution_status") == "failed"]
    terminal_count = len(successes) + len(failures)
    if args.stage == "dev-smoke": st["dev_smoke_complete"] = len(successes) == 3
    elif args.stage == "dev": st["dev_complete"] = len(successes) == len(tasks)
    else:
        st[f"{args.stage}_complete"] = len(successes) == len(tasks)
        st[f"{args.stage}_attempted_count"] = terminal_count
        st[f"{args.stage}_success_count"] = len(successes)
        st[f"{args.stage}_failure_count"] = len(failures)
    save(st, state_path)
    manifest = run_finalize(
        split,
        output / split,
        jsonl_path=output / "per_task_results.jsonl",
        csv_path=output / "results.csv",
    )
    print(json.dumps({
        "stage": args.stage,
        "tasks": len(results),
        "successful_tasks": len(successes),
        "failed_tasks": len(failures),
        "tokens": sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in results),
        "cost": sum(r.get("estimated_cost", 0.0) for r in results),
        "model": provider.model,
        "final_manifest": manifest,
    }, indent=2))


if __name__ == "__main__":
    main()
