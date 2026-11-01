from __future__ import annotations

import hashlib
import json
from pathlib import Path

from freeze_v2_benchmark import verify_manifest

ROOT = Path(__file__).resolve().parent
PROMPT = ROOT / "configs/prompts/direct_mllm_v1.txt"
OUTPUT = ROOT / "outputs/v2/direct_mllm_prompt_freeze.json"
STATE = ROOT / "outputs/v2/direct_mllm/run_state.json"


def prompt_hash() -> str:
    return hashlib.sha256(PROMPT.read_bytes()).hexdigest()


def verify_prompt_freeze(require_frozen: bool = True) -> tuple[bool, list[str]]:
    if not OUTPUT.exists():
        return False, ["prompt freeze record is missing"]
    record = json.loads(OUTPUT.read_text())
    errors = []
    if record["prompt_sha256"] != prompt_hash():
        errors.append("direct_mllm_v1 prompt hash changed")
    if require_frozen and record["status"] != "frozen_after_dev":
        errors.append(f"prompt is not test-frozen: {record['status']}")
    return not errors, errors


def main() -> None:
    benchmark_valid, benchmark_errors = verify_manifest()
    if not benchmark_valid:
        raise SystemExit(f"benchmark freeze violation: {benchmark_errors}")
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    status = "frozen_after_dev" if state.get("dev_complete") else "blocked_not_frozen"
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text())
        if existing.get("status") == "frozen_after_dev" and existing["prompt_sha256"] != prompt_hash():
            raise SystemExit("refusing to modify frozen direct_mllm_v1 prompt")
    benchmark = json.loads((ROOT / "outputs/v2/benchmark_freeze_manifest.json").read_text())
    record = {
        "prompt_version": "direct_mllm_v1",
        "prompt_path": "configs/prompts/direct_mllm_v1.txt",
        "prompt_sha256": prompt_hash(),
        "status": status,
        "dev_complete": bool(state.get("dev_complete")),
        "benchmark_hashes": {
            name: value["aggregate_sha256"] for name, value in benchmark["targets"].items()
        },
    }
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
