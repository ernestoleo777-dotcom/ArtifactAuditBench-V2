from __future__ import annotations

import base64
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from jsonschema import ValidationError, validate

from freeze_v2_benchmark import verify_manifest
from src.providers import OpenRouterProvider

ALLOWED_FILES = {"protocol.json", "observations.csv", "results.csv", "visual.png", "statement.txt"}
FORBIDDEN_TOKENS = ("evaluator_view", "truth", "reference", "canonical", "injector", "clean", "corrupt")


class InputBoundaryError(PermissionError):
    pass


@dataclass
class PreparedInput:
    task_id: str
    messages: list[dict]
    read_files: list[str]
    cache_key: str
    image_bytes: bytes


def stable_csv_text(path: Path) -> str:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()


def parse_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return json.loads(stripped.strip())


class DirectMLLMRunner:
    def __init__(self, root: Path, provider: OpenRouterProvider | None = None, cache_dir: Path | None = None):
        self.root = root.resolve()
        self.provider = provider
        self.cache_dir = cache_dir or (self.root / "outputs/v2/direct_mllm/cache")
        self.prompt_path = self.root / "configs/prompts/direct_mllm_v1.txt"
        self.schema = json.loads((self.root / "configs/v2/direct_mllm_result_schema.json").read_text())
        self.prompt = self.prompt_path.read_text()

    @staticmethod
    def _append_execution_failure(output_dir: Path, record: dict) -> None:
        path = output_dir.parent.parent / "execution_failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_failure(
        self,
        output_dir: Path,
        prepared: PreparedInput,
        envelope: dict,
        *,
        failure_stage: str,
        exc: Exception,
        repaired_format: bool,
    ) -> dict:
        failure = {
            "task_id": prepared.task_id,
            "execution_status": "failed",
            "failure_stage": failure_stage,
            "error_class": type(exc).__name__,
            "error_message": str(exc)[:500],
            "format_repair_used": repaired_format,
            "input_tokens": int(envelope.get("input_tokens", 0)),
            "output_tokens": int(envelope.get("output_tokens", 0)),
            "estimated_cost": float(envelope.get("estimated_cost", 0.0)),
            "latency_seconds": float(envelope.get("latency_seconds", 0.0)),
            "model": str(envelope.get("model", self.provider.model if self.provider else "")),
            "prompt_version": "direct_mllm_v1",
            "read_files": prepared.read_files,
            "cache_key": prepared.cache_key,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        (output_dir / "request_meta.json").write_text(json.dumps({
            "task_id": prepared.task_id,
            "read_files": prepared.read_files,
            "cache_key": prepared.cache_key,
            "format_repair_used": repaired_format,
            "failure_stage": failure_stage,
        }, indent=2, sort_keys=True) + "\n")
        self._append_execution_failure(output_dir, failure)
        return failure

    def _guard_case(self, case_dir: Path) -> Path:
        case = case_dir.resolve()
        try:
            relative = case.relative_to(self.root / "data_v2/agent_view")
        except ValueError as exc:
            raise InputBoundaryError("Direct MLLM can only read data_v2/agent_view") from exc
        if len(relative.parts) != 2 or relative.parts[0] not in {"dev", "test_seen", "test_unseen"}:
            raise InputBoundaryError("invalid Agent-view case path")
        names = {path.name for path in case.iterdir()}
        if names != ALLOWED_FILES:
            raise InputBoundaryError(f"unexpected Agent-view files: {sorted(names ^ ALLOWED_FILES)}")
        return case

    def prepare_input(self, case_dir: Path, model: str) -> PreparedInput:
        case = self._guard_case(case_dir)
        read_files = []
        protocol_path = case / "protocol.json"
        protocol_text = json.dumps(json.loads(protocol_path.read_text()), indent=2, sort_keys=True)
        read_files.append(protocol_path.name)
        protocol = json.loads(protocol_text)
        raw_text = stable_csv_text(case / "observations.csv")
        read_files.append("observations.csv")
        table_text = stable_csv_text(case / "results.csv")
        read_files.append("results.csv")
        claim_text = (case / "statement.txt").read_text()
        read_files.append("statement.txt")
        image_bytes = (case / "visual.png").read_bytes()
        read_files.append("visual.png")
        combined = "\n".join([protocol_text, raw_text, table_text, claim_text]).lower()
        if any(token in combined for token in FORBIDDEN_TOKENS):
            raise InputBoundaryError("forbidden answer-related token in Agent input")
        image_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        required_shape = {
            "task_id": protocol["case_id"],
            "predicted_errors": [{
                "error_type": "allowed error type",
                "affected_artifact": "filename",
                "affected_elements": ["specific element"],
                "reported_values": {},
                "expected_values": {},
                "evidence": ["specific evidence"],
                "confidence": 0.0,
            }],
            "overall_status": "clean|error|unknown",
            "repair_suggestions": [{
                "affected_artifact": "filename",
                "affected_elements": ["element"],
                "suggested_action": "concise correction",
            }],
        }
        user_text = (
            f"TASK ID\n{protocol['case_id']}\n\n"
            f"EXPERIMENT PROTOCOL AND METRIC DEFINITIONS\n{protocol_text}\n\n"
            f"RAW EXPERIMENTAL RESULTS\n```csv\n{raw_text}```\n\n"
            f"REPORTED TABLE\n```csv\n{table_text}```\n\n"
            f"REPORTED CLAIM\n{claim_text}\n\n"
            f"REQUIRED JSON CONTENT SHAPE\n{json.dumps(required_shape, indent=2)}"
        )
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ]},
        ]
        digest = hashlib.sha256()
        digest.update(model.encode())
        digest.update(self.prompt.encode())
        digest.update(user_text.encode())
        digest.update(image_bytes)
        return PreparedInput(protocol["case_id"], messages, read_files, digest.hexdigest(), image_bytes)

    def validate_result(self, result: dict) -> None:
        validate(result, self.schema)

    def dry_run_case(self, case_dir: Path, model: str) -> dict:
        prepared = self.prepare_input(case_dir, model)
        image_part = prepared.messages[1]["content"][1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        return {
            "task_id": prepared.task_id,
            "read_files": prepared.read_files,
            "cache_key": prepared.cache_key,
            "image_bytes": len(prepared.image_bytes),
            "message_count": len(prepared.messages),
            "valid": True,
        }

    def run_case(self, case_dir: Path, output_dir: Path, *, resume: bool = True) -> dict:
        if not self.provider:
            raise RuntimeError("provider is required for non-dry-run execution")
        valid, errors = verify_manifest()
        if not valid:
            raise RuntimeError(f"benchmark freeze violation: {errors}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "result.json"
        prepared = self.prepare_input(case_dir, self.provider.model)
        if resume and result_path.exists():
            meta_path = output_dir / "request_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("cache_key") == prepared.cache_key:
                    cached_result = json.loads(result_path.read_text())
                    self.validate_result(cached_result)
                    return cached_result
        cache_dir = self.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{prepared.cache_key}.json"
        started = time.perf_counter()
        if cache_path.exists():
            envelope = json.loads(cache_path.read_text())
        else:
            response = self.provider.complete(prepared.messages)
            envelope = {
                "text": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "estimated_cost": response.cost,
                "latency_seconds": response.latency_seconds,
                "model": self.provider.model,
            }
            cache_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        repaired_format = False
        try:
            payload = parse_json_object(envelope["text"])
            self._normalize(payload, prepared.task_id, envelope)
            self.validate_result(payload)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError):
            repaired_format = True
            try:
                repair_response = self.provider.repair_json_format(envelope["text"], self.schema)
            except Exception as exc:
                return self._write_failure(
                    output_dir, prepared, envelope,
                    failure_stage="format_repair_request", exc=exc, repaired_format=True
                )
            envelope["input_tokens"] += repair_response.input_tokens
            envelope["output_tokens"] += repair_response.output_tokens
            envelope["estimated_cost"] += repair_response.cost
            envelope["latency_seconds"] += repair_response.latency_seconds
            try:
                payload = parse_json_object(repair_response.text)
                self._normalize(payload, prepared.task_id, envelope)
                self.validate_result(payload)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                return self._write_failure(
                    output_dir, prepared, envelope,
                    failure_stage="format_repair_parse", exc=exc, repaired_format=True
                )
        payload["_format_repaired"] = repaired_format
        payload.pop("_format_repaired")
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        audit_meta = {
            "task_id": prepared.task_id,
            "read_files": prepared.read_files,
            "cache_key": prepared.cache_key,
            "wall_time_seconds": time.perf_counter() - started,
            "format_repair_used": repaired_format,
        }
        (output_dir / "request_meta.json").write_text(json.dumps(audit_meta, indent=2, sort_keys=True) + "\n")
        valid, errors = verify_manifest()
        if not valid:
            raise RuntimeError(f"benchmark changed during run: {errors}")
        return payload

    @staticmethod
    def _normalize(payload: dict, task_id: str, envelope: dict) -> None:
        payload["task_id"] = task_id
        payload["execution_status"] = "success"
        payload["latency_seconds"] = float(envelope["latency_seconds"])
        payload["input_tokens"] = int(envelope["input_tokens"])
        payload["output_tokens"] = int(envelope["output_tokens"])
        payload["estimated_cost"] = float(envelope["estimated_cost"])
        payload["model"] = str(envelope["model"])
        payload["prompt_version"] = "direct_mllm_v1"
