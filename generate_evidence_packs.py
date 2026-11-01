from __future__ import annotations

import hashlib
import json
from pathlib import Path

from freeze_phase4b import verify
from src.tools.evidence_pack import build

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/v2/single_pass_tool/evidence_packs"


def main() -> None:
    valid, errors = verify()
    if not valid:
        raise SystemExit(f"freeze violation: {errors}")
    records = []
    for split in ("dev", "test_seen", "test_unseen"):
        for case in sorted((ROOT / "data_v2/agent_view" / split).iterdir()):
            if not case.is_dir():
                continue
            pack = build(case, ROOT)
            path = OUTPUT / split / f"{case.name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")
            records.append({
                "split": split, "task_id": case.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "path": str(path.relative_to(ROOT)),
            })
    manifest = {"version": "1.0.0", "task_count": len(records), "packs": records}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    valid, errors = verify()
    if not valid:
        raise SystemExit(f"freeze changed during evidence generation: {errors}")
    print(json.dumps({"task_count": len(records), "valid": True}, indent=2))


if __name__ == "__main__":
    main()
