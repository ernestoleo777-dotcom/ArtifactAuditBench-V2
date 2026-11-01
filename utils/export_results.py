"""Export per_task_results.jsonl (the source of truth) to a derived, human-readable results.csv.

This module has no role in computing or altering metrics. It only flattens
already-computed per-task rows into a CSV for paper tables. per_task_results.jsonl
remains authoritative; results.csv is a disposable, regenerable view of it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _flatten(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def export_results(jsonl_path: Path, csv_path: Path) -> int:
    """Write csv_path from jsonl_path. Returns the number of rows written.

    Raises FileNotFoundError if jsonl_path (the source of truth) does not exist;
    this function never fabricates rows for a missing or partial source file.
    """
    jsonl_path, csv_path = Path(jsonl_path), Path(csv_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"source of truth missing, cannot export: {jsonl_path}")
    rows = read_jsonl(jsonl_path)
    flattened = [{key: _flatten(value) for key, value in row.items()} for row in rows]
    frame = pd.DataFrame(flattened)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive results.csv from per_task_results.jsonl (source of truth)."
    )
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    count = export_results(args.jsonl, args.csv)
    print(json.dumps({"rows_exported": count, "csv": str(args.csv)}))


if __name__ == "__main__":
    main()
