"""Deterministic post-run finalize step and runtime-failure logging.

Two independent concerns live here, deliberately kept separate from
evaluation semantics:

- append_execution_failure: an append-only log of runtime/IO/parsing
  failures (e.g. JSONDecodeError) that occur while *executing* a task.
  This is unrelated to evaluate_single_pass_tool.py's failure_cases.jsonl,
  which records semantic truth-vs-prediction mismatches. The two files
  must never be merged.

- run_finalize: counts what is already on disk for one stage (split or
  ablation variant) and writes a manifest. It never recomputes metrics,
  never re-runs the model, and never fabricates rows for missing tasks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from utils.export_results import export_results


def append_execution_failure(
    path: Path,
    *,
    task_id: str,
    stage: str,
    error_type: str,
    error_message: str,
    raw_output: str | None = None,
) -> None:
    """Append one runtime-failure record to path. Never raises.

    Logging must never block or crash the pipeline, so any IO problem
    while writing this diagnostic record is swallowed rather than propagated.
    """
    record = {
        "task_id": task_id,
        "stage": stage,
        "error_type": error_type,
        "error_message": str(error_message)[:500],
        "raw_output": raw_output[:2000] if isinstance(raw_output, str) else raw_output,
        "timestamp": time.time(),
    }
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def _jsonl_row_count_for_stage(jsonl_path: Path, stage: str) -> int | None:
    if not jsonl_path.exists():
        return None
    count = 0
    with jsonl_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get("split") == stage:
                count += 1
    return count


def run_finalize(
    stage: str,
    output_root: Path,
    *,
    jsonl_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict:
    """Validate and summarize one stage's already-completed task outputs.

    output_root must be the per-stage directory (e.g.
    outputs/v2/single_pass_tool/test_seen) holding one subdirectory per
    task_id, each containing result.json (success) or failure.json
    (execution failure). This function only counts what is already
    written; it does not call the model or alter any result.

    If jsonl_path (the method-level source-of-truth per_task_results.jsonl)
    already exists and csv_path does not, the derived results.csv is
    generated from it. Neither file is required for the manifest to be
    written -- a stage can be finalized before evaluation has run.
    """
    output_root = Path(output_root)
    task_dirs = sorted(p for p in output_root.iterdir() if p.is_dir()) if output_root.exists() else []
    success = [p.name for p in task_dirs if (p / "result.json").exists()]
    failed = [p.name for p in task_dirs if (p / "failure.json").exists() and not (p / "result.json").exists()]
    incomplete = [
        p.name for p in task_dirs
        if not (p / "result.json").exists() and not (p / "failure.json").exists()
    ]

    csv_generated = False
    if jsonl_path is not None and csv_path is not None:
        jsonl_path, csv_path = Path(jsonl_path), Path(csv_path)
        if jsonl_path.exists() and not csv_path.exists():
            export_results(jsonl_path, csv_path)
            csv_generated = True

    manifest = {
        "stage": stage,
        "task_count": len(task_dirs),
        "success_count": len(success),
        "failure_count": len(failed),
        "incomplete_count": len(incomplete),
        "incomplete_task_ids": incomplete,
        "jsonl_source_of_truth": str(jsonl_path) if jsonl_path else None,
        "jsonl_exists": bool(jsonl_path and jsonl_path.exists()),
        "jsonl_row_count_for_stage": _jsonl_row_count_for_stage(jsonl_path, stage) if jsonl_path else None,
        "csv_path": str(csv_path) if csv_path else None,
        "csv_exists": bool(csv_path and csv_path.exists()),
        "csv_generated_this_run": csv_generated,
        "timestamp": time.time(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "final_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
