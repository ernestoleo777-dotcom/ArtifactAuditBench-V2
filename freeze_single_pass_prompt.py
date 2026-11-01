from __future__ import annotations

import hashlib
import json
from pathlib import Path

from freeze_phase4b import verify

ROOT = Path(__file__).resolve().parent
PROMPT = ROOT / "configs/prompts/single_pass_tool_v1.txt"
OUTPUT = ROOT / "outputs/v2/single_pass_tool/prompt_freeze.json"
STATE = ROOT / "outputs/v2/single_pass_tool/run_state.json"


def prompt_hash(): return hashlib.sha256(PROMPT.read_bytes()).hexdigest()


def verify_prompt(require_frozen=True):
    if not OUTPUT.exists(): return False, ["prompt freeze missing"]
    record = json.loads(OUTPUT.read_text())
    errors = []
    if record["prompt_sha256"] != prompt_hash(): errors.append("single-pass prompt changed")
    if require_frozen and record["status"] != "frozen_after_dev": errors.append("prompt not frozen after dev")
    return not errors, errors


def main():
    valid, errors = verify()
    if not valid: raise SystemExit(f"freeze violation: {errors}")
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    status = "frozen_after_dev" if state.get("dev_complete") else "blocked_not_frozen"
    if OUTPUT.exists():
        old = json.loads(OUTPUT.read_text())
        if old.get("status") == "frozen_after_dev" and old["prompt_sha256"] != prompt_hash():
            raise SystemExit("refusing to modify frozen single_pass_tool_v1")
    record = {"prompt_version": "single_pass_tool_v1", "prompt_sha256": prompt_hash(), "status": status, "dev_complete": bool(state.get("dev_complete"))}
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
