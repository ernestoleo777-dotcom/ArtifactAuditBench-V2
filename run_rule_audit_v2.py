from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

from src.v2.baseline import audit_case

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "data_v2"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/v2/rule_audit"


def load_cases(data_root: Path, splits: list[str], manifest: Path | None) -> dict[str, list[Path]]:
    if manifest:
        data = json.loads(manifest.read_text())
        return {
            split: [data_root / "agent_view" / split / case_id for case_id in data["splits"].get(split, [])]
            for split in splits
        }
    return {
        split: sorted(path for path in (data_root / "agent_view" / split).iterdir() if path.is_dir())
        for split in splits
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--splits", default="dev,test_seen,test_unseen")
    parser.add_argument("--manifest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output_root).resolve()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    manifest = Path(args.manifest).resolve() if args.manifest else None
    cases_by_split = load_cases(data_root, splits, manifest)
    if args.dry_run:
        summary = {
            "status": "dry_run",
            "data_root": str(data_root),
            "output_root": str(output),
            "task_count": sum(len(cases) for cases in cases_by_split.values()),
            "by_split": {split: len(cases) for split, cases in cases_by_split.items()},
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    results = []
    for split, cases in cases_by_split.items():
        for case in cases:
            if case.is_dir():
                results.append((split, audit_case(case, output / split / case.name, ROOT)))
    summary = {
        "task_count": len(results),
        "by_split": dict(Counter(split for split, _ in results)),
        "predicted_error_instances": dict(Counter(
            item["error_type"] for _, result in results for item in result["predicted_errors"]
        )),
        "unknown_tasks": sum(bool(result["unknown"]) for _, result in results),
        "mean_execution_time_seconds": (
            sum(result["execution_time_seconds"] for _, result in results) / len(results) if results else 0.0
        ),
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
