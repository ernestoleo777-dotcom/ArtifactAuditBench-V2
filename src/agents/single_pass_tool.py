from __future__ import annotations

import base64
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from jsonschema import ValidationError, validate

from freeze_phase4b import verify as verify_freeze
from src.agents.direct_mllm import parse_json_object
from src.providers import OpenRouterProvider
from utils.finalize import append_execution_failure

VERSION = "1.0.0"


def safe_response_summary(text: str) -> dict:
    encoded = text.encode("utf-8", errors="replace")
    stripped = text.strip()
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "character_count": len(text),
        "byte_count": len(encoded),
        "starts_with_json_object": stripped.startswith("{"),
        "ends_with_json_object": stripped.endswith("}"),
        "contains_markdown_fence": "```" in text,
    }


def csv_text(path: Path) -> str:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    out = StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


def evidence_variant(pack: dict, variant: str) -> dict:
    base = {
        "task_id": pack["task_id"],
        "metric_facts": pack["metric_facts"],
        "ranking_facts": pack["ranking_facts"],
        "claim_text": pack["claim_text"],
        "tool_uncertainties": pack["tool_uncertainties"],
        "tool_versions": pack["tool_versions"],
    }
    if variant in {"structured", "full"}:
        base.update({
            "reported_table_facts": pack["reported_table_facts"],
            "numeric_discrepancies": pack["numeric_discrepancies"],
            "aggregation_facts": pack["aggregation_facts"],
            "statistical_facts": pack["statistical_facts"],
        })
    if variant == "full":
        base["figure_facts"] = pack["figure_facts"]
    return base


class SinglePassToolRunner:
    def __init__(
        self,
        root: Path,
        provider: OpenRouterProvider | None = None,
        cache_dir: Path | None = None,
        evidence_root: Path | None = None,
        enforce_phase4b_freeze: bool = True,
    ):
        self.root = root.resolve()
        self.provider = provider
        self.cache_dir = cache_dir or (self.root / "outputs/v2/single_pass_tool/cache")
        self.evidence_root = evidence_root or (self.root / "outputs/v2/single_pass_tool/evidence_packs")
        self.enforce_phase4b_freeze = enforce_phase4b_freeze
        self.execution_failures_path = self.cache_dir.parent / "execution_failures.jsonl"
        self.prompt = (root / "configs/prompts/single_pass_tool_v1.txt").read_text()
        self.schema_path = root / "configs/v2/single_pass_tool_result_schema.json"
        self.schema = json.loads(self.schema_path.read_text())
        self.schema_hash = hashlib.sha256(self.schema_path.read_bytes()).hexdigest()

    def prepare(self, split: str, task_id: str, model: str, variant: str = "full") -> dict:
        case = self.root / "data_v2/agent_view" / split / task_id
        pack_path = self.evidence_root / split / f"{task_id}.json"
        if not case.is_dir() or not pack_path.exists():
            raise PermissionError("single-pass input must be an existing agent_view case and evidence pack")
        protocol = json.loads((case / "protocol.json").read_text())
        claim = (case / "statement.txt").read_text()
        table = csv_text(case / "results.csv")
        image = (case / "visual.png").read_bytes()
        pack = json.loads(pack_path.read_text())
        selected = evidence_variant(pack, variant)
        required_shape = {
            "task_id": task_id,
            "predicted_errors": [{
                "error_type": "allowed_type",
                "affected_artifact": "filename",
                "affected_elements": ["element"],
                "reported_values": {},
                "expected_values": {},
                "evidence": ["short fact"],
                "confidence": 0.0
            }],
            "overall_status": "clean|error|unknown",
            "repair_suggestions": [{
                "affected_artifact": "filename",
                "affected_elements": ["element"],
                "suggested_action": "short correction"
            }]
        }
        user_text = (
            f"TASK ID\n{task_id}\n\n"
            f"EXPERIMENT PROTOCOL\n{json.dumps(protocol, indent=2, sort_keys=True)}\n\n"
            f"REPORTED TABLE\n```csv\n{table}```\n\n"
            f"REPORTED CLAIM\n{claim}\n\n"
            f"DETERMINISTIC TOOL EVIDENCE ({variant})\n{json.dumps(selected, separators=(',', ':'), sort_keys=True)}\n\n"
            f"REQUIRED JSON CONTENT SHAPE\n{json.dumps(required_shape, separators=(',', ':'))}\n"
            "Return only this compact JSON object. Omit an error rather than adding long explanations."
        )
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}},
            ]},
        ]
        evidence_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
        image_hash = hashlib.sha256(image).hexdigest()
        prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()
        max_tokens = getattr(self.provider, "max_tokens", 0) if self.provider else 0
        cache_key = hashlib.sha256(
            f"{model}|{prompt_hash}|{evidence_hash}|{image_hash}|{self.schema_hash}|{variant}|{max_tokens}".encode()
        ).hexdigest()
        return {
            "task_id": task_id, "messages": messages, "cache_key": cache_key,
            "evidence_hash": evidence_hash, "image_hash": image_hash, "prompt_hash": prompt_hash,
            "read_files": ["protocol.json", "results.csv", "statement.txt", "visual.png", f"evidence_packs/{split}/{task_id}.json"],
            "tool_versions": pack["tool_versions"], "variant": variant,
            "max_output_tokens": max_tokens,
        }

    def run(self, split: str, task_id: str, output_dir: Path, variant="full", resume=True) -> dict:
        if not self.provider: raise RuntimeError("provider required")
        if self.enforce_phase4b_freeze:
            valid, errors = verify_freeze()
            if not valid: raise RuntimeError(f"freeze violation: {errors}")
        prepared = self.prepare(split, task_id, self.provider.model, variant)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path, meta_path = output_dir / "result.json", output_dir / "request_meta.json"
        if resume and result_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("cache_key") == prepared["cache_key"]:
                result = json.loads(result_path.read_text()); validate(result, self.schema); return result
        cache = self.cache_dir / f"{prepared['cache_key']}.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        if cache.exists():
            envelope = json.loads(cache.read_text())
        else:
            response = self.provider.complete(prepared["messages"])
            envelope = {
                "text": response.text, "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens, "estimated_cost": response.cost,
                "latency_seconds": response.latency_seconds, "model": self.provider.model,
            }
            cache.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        repaired = False
        try:
            result = parse_json_object(envelope["text"])
            self._coerce_structure(result)
            self._normalize(result, prepared, envelope)
            validate(result, self.schema)
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as original_exc:
            repaired = True
            append_execution_failure(
                self.execution_failures_path,
                task_id=task_id,
                stage=split,
                error_type=type(original_exc).__name__,
                error_message=str(original_exc),
                raw_output=envelope["text"],
            )
            try:
                repair = self.provider.repair_json_format(envelope["text"], self.schema)
            except Exception as repair_request_exc:
                append_execution_failure(
                    self.execution_failures_path,
                    task_id=task_id,
                    stage=split,
                    error_type=type(repair_request_exc).__name__,
                    error_message=str(repair_request_exc),
                    raw_output=envelope["text"],
                )
                return self._write_failure(
                    split=split,
                    output_dir=output_dir,
                    prepared=prepared,
                    envelope=envelope,
                    started=start,
                    failure_stage="format_repair_request",
                    original_exception=original_exc,
                    final_exception=repair_request_exc,
                    repair_text=None,
                )
            envelope["input_tokens"] += repair.input_tokens
            envelope["output_tokens"] += repair.output_tokens
            envelope["estimated_cost"] += repair.cost
            envelope["latency_seconds"] += repair.latency_seconds
            try:
                result = parse_json_object(repair.text)
                self._coerce_structure(result)
                self._normalize(result, prepared, envelope)
                validate(result, self.schema)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as final_exc:
                append_execution_failure(
                    self.execution_failures_path,
                    task_id=task_id,
                    stage=split,
                    error_type=type(final_exc).__name__,
                    error_message=str(final_exc),
                    raw_output=repair.text,
                )
                return self._write_failure(
                    split=split,
                    output_dir=output_dir,
                    prepared=prepared,
                    envelope=envelope,
                    started=start,
                    failure_stage="format_repair_parse",
                    original_exception=original_exc,
                    final_exception=final_exc,
                    repair_text=repair.text,
                )
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        failure_path = output_dir / "failure.json"
        if failure_path.exists():
            failure_path.unlink()
        meta = {
            "task_id": task_id, "cache_key": prepared["cache_key"],
            "prompt_hash": prepared["prompt_hash"], "evidence_hash": prepared["evidence_hash"],
            "image_hash": prepared["image_hash"], "schema_hash": self.schema_hash,
            "variant": variant, "read_files": prepared["read_files"],
            "max_output_tokens": prepared["max_output_tokens"],
            "model_analysis_calls": 1, "format_repair_calls": int(repaired),
            "wall_time_seconds": time.perf_counter() - start,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        if self.enforce_phase4b_freeze:
            valid, errors = verify_freeze()
            if not valid: raise RuntimeError(f"freeze changed during run: {errors}")
        return result

    def _write_failure(
        self,
        *,
        split: str,
        output_dir: Path,
        prepared: dict,
        envelope: dict,
        started: float,
        failure_stage: str,
        original_exception: Exception,
        final_exception: Exception,
        repair_text: str | None,
    ) -> dict:
        failure_type = {
            "format_repair_parse": "double_json_parse_failure",
            "format_repair_request": "format_repair_request_failure",
        }.get(failure_stage, failure_stage)
        failure = {
            "task_id": prepared["task_id"],
            "split": split,
            "variant": prepared["variant"],
            "model": str(envelope["model"]),
            "status": "failed",
            "execution_status": "failed",
            "failure_stage": failure_stage,
            "failure_type": failure_type,
            "first_parse_error": {
                "type": type(original_exception).__name__,
                "summary": str(original_exception)[:300],
            },
            "repair_parse_error": (
                {"type": type(final_exception).__name__, "summary": str(final_exception)[:300]}
                if failure_stage == "format_repair_parse"
                else None
            ),
            "original_exception_type": type(original_exception).__name__,
            "final_exception_type": type(final_exception).__name__,
            "original_response_summary": safe_response_summary(envelope["text"]),
            "repair_response_summary": safe_response_summary(repair_text) if repair_text is not None else None,
            "input_tokens": int(envelope["input_tokens"]),
            "output_tokens": int(envelope["output_tokens"]),
            "estimated_cost": float(envelope["estimated_cost"]),
            "latency_seconds": float(envelope["latency_seconds"]),
            "wall_time_seconds": time.perf_counter() - started,
            "cache_key": prepared["cache_key"],
            "prompt_hash": prepared["prompt_hash"],
            "prompt_version": "single_pass_tool_v1",
            "tool_versions": prepared["tool_versions"],
            "read_files": prepared["read_files"],
            "predictions_recorded": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (output_dir / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return failure

    @staticmethod
    def _coerce_structure(result: dict):
        status = result.get("overall_status")
        if status in {"unclean", "issues_found", "inconsistent"}:
            result["overall_status"] = "error"
        result.setdefault("predicted_errors", [])
        result.setdefault("repair_suggestions", [])
        normalized = []
        for item in result["predicted_errors"][:3]:
            evidence = item.get("evidence", item.get("explanation", []))
            if isinstance(evidence, str):
                evidence = [evidence]
            normalized.append({
                "error_type": item.get("error_type", "unknown"),
                "affected_artifact": item.get("affected_artifact", "unknown"),
                "affected_elements": item.get("affected_elements", []),
                "reported_values": item.get("reported_values", {}),
                "expected_values": item.get("expected_values", {}),
                "evidence": evidence[:3],
                "confidence": float(item.get("confidence", 0.5)),
            })
        result["predicted_errors"] = normalized
        if result.get("overall_status") == "clean":
            result["predicted_errors"] = []

    @staticmethod
    def _normalize(result: dict, prepared: dict, envelope: dict):
        result["task_id"] = prepared["task_id"]
        result["execution_status"] = "success"
        result["latency_seconds"] = float(envelope["latency_seconds"])
        result["input_tokens"] = int(envelope["input_tokens"])
        result["output_tokens"] = int(envelope["output_tokens"])
        result["estimated_cost"] = float(envelope["estimated_cost"])
        result["model"] = str(envelope["model"])
        result["prompt_version"] = "single_pass_tool_v1"
        result["tool_versions"] = prepared["tool_versions"]
