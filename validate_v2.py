from __future__ import annotations

import hashlib
import json
import argparse
from collections import Counter
from pathlib import Path

from jsonschema import validate
from PIL import Image

from src.v2.recompute import expected_table, read_csv, recompute_summary

ROOT = Path(__file__).resolve().parent
FORBIDDEN_AGENT_TERMS = [
    "ground_truth", "truth.json", "reference_", "canonical_facts", "injector",
    "error_type", "clean", "corrupted", "template_fingerprint", "style_fingerprint",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data_v2"))
    parser.add_argument("--manifest")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--strict-v2-120", action="store_true")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else data_root / "split_manifest.json"
    strict_v2 = args.strict_v2_120 or data_root.name == "data_v2"
    manifest = json.loads(manifest_path.read_text())
    schema = json.loads((ROOT / "configs/v2/task_schema.json").read_text())
    if not strict_v2:
        schema = json.loads(json.dumps(schema))
        schema.get("properties", {}).get("split", {})["enum"] = list(manifest["splits"])
    errors = []
    records = []
    for split, case_ids in manifest["splits"].items():
        for cid in case_ids:
            agent = data_root / "agent_view" / split / cid
            evaluator = data_root / "evaluator_view" / split / cid
            try:
                assert {p.name for p in agent.iterdir()} == {
                    "observations.csv", "protocol.json", "results.csv", "visual.png", "statement.txt"
                }
                combined = "\n".join(
                    p.read_text(errors="ignore") for p in agent.iterdir() if p.suffix in {".json", ".txt", ".csv"}
                ).lower()
                assert not any(term in combined for term in FORBIDDEN_AGENT_TERMS)
                task = json.loads((evaluator / "task.json").read_text())
                validate(task, schema)
                assert task["case_id"] == cid and task["split"] == split
                protocol = json.loads((agent / "protocol.json").read_text())
                summary = recompute_summary(read_csv(agent / "observations.csv"), protocol)
                facts = json.loads((evaluator / "canonical_facts.json").read_text())
                assert summary.keys() == facts["summary"].keys()
                expected = expected_table(summary, protocol["experiment_type"])
                reference = read_csv(evaluator / "reference_table.csv")
                assert expected == reference
                image = Image.open(agent / "visual.png").convert("RGB")
                assert image.width > 0 and image.height > 0 and any(a != b for a, b in image.getextrema())
                truth = json.loads((evaluator / "truth.json").read_text())
                assert len(truth["errors"]) == len(task["errors"])
                assert all(item["before"] != item["after"] for item in truth["errors"])
                records.append(task)
            except Exception as exc:
                errors.append(f"{split}/{cid}: {exc}")
    try:
        if args.expected_count is not None:
            assert len(records) == args.expected_count
        if strict_v2:
            assert len(records) == 120
            assert Counter(len(r["errors"]) for r in records) == {0: 30, 1: 60, 2: 21, 3: 9}
        seed_sets = {split: {r["seed"] for r in records if r["split"] == split} for split in manifest["splits"]}
        template_sets = {split: set(manifest["isolation"]["template_fingerprints"][split]) for split in manifest["splits"]}
        style_sets = {split: set(manifest["isolation"]["style_fingerprints"][split]) for split in manifest["splits"]}
        split_names = list(manifest["splits"])
        for i, left in enumerate(split_names):
            for right in split_names[i + 1:]:
                assert seed_sets[left].isdisjoint(seed_sets[right])
                if strict_v2:
                    assert template_sets[left].isdisjoint(template_sets[right])
                    assert style_sets[left].isdisjoint(style_sets[right])
    except Exception as exc:
        errors.append(f"isolation: {exc}")
    result = {"valid": not errors, "task_count": len(records), "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
