from __future__ import annotations

import hashlib
import json
from pathlib import Path

from freeze_v2_benchmark import verify_manifest as verify_v2
from freeze_direct_prompt import verify_prompt_freeze

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/v2/single_pass_tool/freeze_manifest.json"
TARGETS = {
    "v2_freeze": ROOT / "outputs/v2/benchmark_freeze_manifest.json",
    "rule_audit": ROOT / "outputs/v2/rule_audit",
    "rule_evaluation": ROOT / "outputs/v2/rule_audit_evaluation.json",
    "direct_prompt": ROOT / "configs/prompts/direct_mllm_v1.txt",
    "direct_prompt_freeze": ROOT / "outputs/v2/direct_mllm_prompt_freeze.json",
    "direct_results": ROOT / "outputs/v2/direct_mllm",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def files(path: Path) -> dict:
    if path.is_file():
        return {str(path.relative_to(ROOT)): digest(path)}
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def aggregate(items: dict) -> str:
    return hashlib.sha256("\n".join(f"{k}\t{v}" for k, v in sorted(items.items())).encode()).hexdigest()


def current() -> dict:
    result = {}
    for name, path in TARGETS.items():
        inventory = files(path)
        result[name] = {"path": str(path.relative_to(ROOT)), "files": inventory, "aggregate_sha256": aggregate(inventory)}
    return {"version": "1.0.0", "algorithm": "sha256", "targets": result}


def verify(path: Path = OUTPUT) -> tuple[bool, list[str]]:
    if not path.exists():
        return False, ["phase 4B freeze manifest missing"]
    expected, actual = json.loads(path.read_text()), current()
    errors = []
    for name in TARGETS:
        if expected["targets"][name] != actual["targets"][name]:
            errors.append(f"{name} changed")
    v2_ok, v2_errors = verify_v2()
    prompt_ok, prompt_errors = verify_prompt_freeze(True)
    return not errors and v2_ok and prompt_ok, errors + v2_errors + prompt_errors


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        valid, errors = verify()
        print(json.dumps({"valid": valid, "errors": errors}, indent=2))
        raise SystemExit(0 if valid else 1)
    v2_ok, v2_errors = verify_v2()
    prompt_ok, prompt_errors = verify_prompt_freeze(True)
    if not v2_ok or not prompt_ok:
        raise SystemExit(f"existing freeze invalid: {v2_errors + prompt_errors}")
    value = current()
    OUTPUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": True, "hashes": {k: v["aggregate_sha256"] for k, v in value["targets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
