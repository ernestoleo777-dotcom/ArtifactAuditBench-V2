from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from freeze_direct_prompt import verify_prompt_freeze
from freeze_v2_benchmark import verify_manifest
from src.agents import DirectMLLMRunner
from src.providers import OpenRouterProvider, ProviderConfigurationError

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/v2/direct_mllm"
STATE_PATH = OUTPUT / "run_state.json"


def load_config() -> dict:
    return json.loads((ROOT / "configs/models/direct_mllm.json").read_text())


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_").replace(" ", "_")


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def load_manifest(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


def cases_for(split: str, manifest: dict | None = None) -> list[Path]:
    if manifest:
        return [ROOT / "data_v2/agent_view" / split / task_id for task_id in manifest["splits"].get(split, [])]
    return sorted(path for path in (ROOT / "data_v2/agent_view" / split).iterdir() if path.is_dir())


def manifest_cases(manifest: dict | None) -> list[tuple[str, Path]]:
    if not manifest:
        return [(split, case) for split in ("dev", "test_seen", "test_unseen") for case in cases_for(split)]
    return [
        (split, ROOT / "data_v2/agent_view" / split / task_id)
        for split, task_ids in manifest["splits"].items()
        for task_id in task_ids
    ]


def assert_stage_allowed(stage: str, state: dict) -> None:
    if stage not in {"dry-run"} and not state.get("dry_run_complete"):
        raise RuntimeError("dry-run must complete before API execution")
    if stage == "dev" and not state.get("dev_smoke_complete"):
        raise RuntimeError("dev-smoke must complete before full dev")
    if stage in {"test_seen", "test_unseen"}:
        valid, errors = verify_prompt_freeze(require_frozen=True)
        if not valid:
            raise RuntimeError(f"test execution requires frozen prompt: {errors}")
        if not state.get("dev_complete"):
            raise RuntimeError("full dev must complete before test execution")
    if stage == "test_unseen" and not state.get("test_seen_complete"):
        raise RuntimeError("test_seen must complete before test_unseen")
    if stage == "test_seen" and state.get("test_seen_complete"):
        raise RuntimeError("test_seen was already executed for direct_mllm_v1")
    if stage == "test_unseen" and state.get("test_unseen_complete"):
        raise RuntimeError("test_unseen was already executed for direct_mllm_v1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["dry-run", "smoke", "dev-smoke", "dev", "test_seen", "test_unseen"])
    parser.add_argument("--case")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--model")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    benchmark_valid, benchmark_errors = verify_manifest()
    if not benchmark_valid:
        raise SystemExit(f"benchmark freeze violation: {benchmark_errors}")
    config = load_config()
    if args.model:
        os.environ[config["model_env"]] = args.model
    configured_model = os.environ.get(config["model_env"])
    output = Path(args.output_root).resolve() if args.output_root else (
        ROOT / "outputs/v2/model_sweep" / model_slug(configured_model) / "direct_mllm"
        if args.model and configured_model else OUTPUT
    )
    is_model_sweep = (ROOT / "outputs/v2/model_sweep").resolve() in output.parents
    state_path = output / "run_state.json"
    output.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path)
    manifest = load_manifest(args.manifest)
    runner = DirectMLLMRunner(ROOT, cache_dir=output / "cache" if is_model_sweep else None)
    if args.list:
        if args.stage == "smoke":
            selected_pairs = manifest_cases(manifest)
        else:
            split = "dev" if args.stage in {"dev-smoke", "dev", "dry-run"} else args.stage
            selected_pairs = [(split, case) for case in cases_for(split, manifest)]
        if args.stage == "dev-smoke":
            selected_pairs = selected_pairs[:3]
        print(json.dumps({
            "status": "list",
            "stage": args.stage,
            "output_root": str(output),
            "model": configured_model or "<ARTIFACT_AUDIT_MODEL unset>",
            "tasks": [{"split": split, "task_id": case.name} for split, case in selected_pairs],
        }, indent=2, sort_keys=True))
        return
    assert_stage_allowed(args.stage, state)

    if args.stage == "dry-run":
        model_label = configured_model or "<ARTIFACT_AUDIT_MODEL unset>"
        summaries = []
        for split in ("dev", "test_seen", "test_unseen"):
            for case in cases_for(split, manifest):
                summaries.append({"split": split, **runner.dry_run_case(case, model_label)})
        dry_run_output = {
            "status": "success",
            "task_count": len(summaries),
            "model_configured": bool(configured_model),
            "all_visual_inputs_embedded": all(item["image_bytes"] > 0 for item in summaries),
            "tasks": summaries,
        }
        (output / "dry_run.json").write_text(json.dumps(dry_run_output, indent=2, sort_keys=True) + "\n")
        state["dry_run_complete"] = True
        save_state(state, state_path)
        print(json.dumps({key: dry_run_output[key] for key in ("status", "task_count", "model_configured", "all_visual_inputs_embedded")}, indent=2))
        return

    try:
        provider = OpenRouterProvider(config)
    except ProviderConfigurationError as exc:
        blocked = {"status": "blocked", "stage": args.stage, "reason": str(exc)}
        (output / "blocked.json").write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n")
        print(json.dumps(blocked, indent=2))
        raise SystemExit(2)
    runner = DirectMLLMRunner(ROOT, provider, cache_dir=output / "cache" if is_model_sweep else None)
    split = "dev" if args.stage in {"dev-smoke", "dev"} else args.stage
    selected_pairs = manifest_cases(manifest) if args.stage == "smoke" else [(split, case) for case in cases_for(split, manifest)]
    if args.case:
        selected_pairs = [(s, case) for s, case in selected_pairs if case.name == args.case]
        if not selected_pairs:
            raise SystemExit("requested case was not found in the selected split")
    elif args.stage == "dev-smoke":
        selected_pairs = selected_pairs[:3]
    results = []
    for case_split, case in selected_pairs:
        results.append(runner.run_case(case, output / case_split / case.name, resume=not args.no_resume))
    if args.stage == "smoke":
        state["smoke_complete"] = len(results) == len(selected_pairs)
        state["smoke_task_count"] = len(results)
    if args.stage == "dev-smoke" and len(results) == 3:
        state["dev_smoke_complete"] = True
    elif args.stage == "dev" and len(results) == len(cases_for(split, manifest)):
        state["dev_complete"] = True
    elif args.stage == "test_seen" and len(results) == len(cases_for(split, manifest)):
        state["test_seen_complete"] = True
    elif args.stage == "test_unseen" and len(results) == len(cases_for(split, manifest)):
        state["test_unseen_complete"] = True
    save_state(state, state_path)
    benchmark_valid, benchmark_errors = verify_manifest()
    if not benchmark_valid:
        raise SystemExit(f"benchmark changed during execution: {benchmark_errors}")
    print(json.dumps({
        "status": "success", "stage": args.stage, "tasks": len(results),
        "model": provider.model,
        "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in results),
        "cost": sum(r["estimated_cost"] for r in results),
    }, indent=2))


if __name__ == "__main__":
    main()
