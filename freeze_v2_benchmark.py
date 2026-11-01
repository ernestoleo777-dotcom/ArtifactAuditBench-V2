from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/v2/benchmark_freeze_manifest.json"
TARGETS = {
    "split_manifest": ROOT / "data_v2/split_manifest.json",
    "evaluator_view": ROOT / "data_v2/evaluator_view",
    "test_seen": ROOT / "data_v2/agent_view/test_seen",
    "test_unseen": ROOT / "data_v2/agent_view/test_unseen",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, str]:
    if path.is_file():
        return {str(path.relative_to(ROOT)): sha256(path)}
    return {
        str(item.relative_to(ROOT)): sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def aggregate(files: dict[str, str]) -> str:
    payload = "\n".join(f"{name}\t{digest}" for name, digest in sorted(files.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_manifest() -> dict:
    targets = {}
    for name, path in TARGETS.items():
        files = inventory(path)
        targets[name] = {
            "path": str(path.relative_to(ROOT)),
            "file_count": len(files),
            "aggregate_sha256": aggregate(files),
            "files": files,
        }
    return {
        "manifest_version": "1.0.0",
        "algorithm": "sha256",
        "targets": targets,
    }


def verify_manifest(manifest_path: Path = OUTPUT) -> tuple[bool, list[str]]:
    expected = json.loads(manifest_path.read_text())
    current = build_manifest()
    errors = []
    for name in TARGETS:
        if expected["targets"][name]["aggregate_sha256"] != current["targets"][name]["aggregate_sha256"]:
            errors.append(f"{name}: aggregate hash changed")
        if expected["targets"][name]["files"] != current["targets"][name]["files"]:
            errors.append(f"{name}: file inventory changed")
    return not errors, errors


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        valid, errors = verify_manifest()
        print(json.dumps({"frozen": True, "valid": valid, "errors": errors}, indent=2))
        raise SystemExit(0 if valid else 1)
    manifest = build_manifest()
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "frozen": True,
        "valid": True,
        "manifest": str(OUTPUT.relative_to(ROOT)),
        "aggregate_sha256": {
            name: details["aggregate_sha256"] for name, details in manifest["targets"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
