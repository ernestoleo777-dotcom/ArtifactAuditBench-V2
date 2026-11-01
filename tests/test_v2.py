import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from generate_v2 import ROOT
from src.v2.baseline import V2AccessViolation, V2Reader, audit_case
from src.v2.core import ERROR_TYPES, METRIC_DIRECTIONS, run_statistics, significance


DATA = ROOT / "data_v2"
OUTPUT = ROOT / "outputs/v2"


def _records():
    records = []
    manifest = json.loads((DATA / "split_manifest.json").read_text())
    for split, ids in manifest["splits"].items():
        for cid in ids:
            records.append(json.loads((DATA / "evaluator_view" / split / cid / "task.json").read_text()))
    return records


def test_v2_has_120_complete_tasks():
    records = _records()
    assert len(records) == 120
    assert Counter(len(r["errors"]) for r in records) == {0: 30, 1: 60, 2: 21, 3: 9}
    for record in records:
        agent = DATA / record["paths"]["agent"]
        evaluator = DATA / record["paths"]["evaluator"]
        assert agent.exists() and evaluator.exists()
        assert {p.name for p in agent.iterdir()} == {
            "observations.csv", "protocol.json", "results.csv", "visual.png", "statement.txt"
        }


def test_split_seed_template_and_style_isolation():
    records = _records()
    manifest = json.loads((DATA / "split_manifest.json").read_text())
    for field in ("seed", "template_fingerprint", "style_fingerprint"):
        sets = {split: {r[field] for r in records if r["split"] == split} for split in manifest["splits"]}
        assert sets["dev"].isdisjoint(sets["test_seen"])
        assert sets["dev"].isdisjoint(sets["test_unseen"])
        assert sets["test_seen"].isdisjoint(sets["test_unseen"])


def test_combination_ground_truth_and_all_error_types():
    records = _records()
    combos = [r for r in records if len(r["errors"]) >= 2]
    assert len(combos) == 30
    assert set(ERROR_TYPES) == {e["error_type"] for r in records for e in r["errors"]}
    for record in combos:
        assert 2 <= len(record["errors"]) <= 3
        assert len({e["error_type"] for e in record["errors"]}) == len(record["errors"])


def test_all_injectors_are_reversible_in_ground_truth():
    for record in _records():
        for error in record["errors"]:
            assert error["before"] != error["after"]
            assert error["repair"] == error["before"]
            assert error["affected_artifact"]
            assert error["affected_elements"]


def test_multimodel_metric_directions():
    for metric in ("Accuracy", "Precision", "Recall", "Macro-F1", "Micro-F1", "Weighted-F1", "ROC-AUC", "R2"):
        assert METRIC_DIRECTIONS[metric] == "maximize"
    for metric in ("MAE", "RMSE"):
        assert METRIC_DIRECTIONS[metric] == "minimize"
    assert any(json.loads((DATA / r["paths"]["agent"] / "protocol.json").read_text())["model_count"] == 5 for r in _records())


def test_confidence_interval_calculation():
    stats = run_statistics([0.7, 0.72, 0.68, 0.71, 0.69])
    assert stats["mean"] == pytest.approx(0.7)
    assert stats["std"] == pytest.approx(np.std([0.7, 0.72, 0.68, 0.71, 0.69], ddof=1))
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]
    assert stats["se"] == pytest.approx(stats["std"] / np.sqrt(5))


def test_significance_logic():
    overlap = {
        "A": {"ci95_low": 0.70, "ci95_high": 0.75},
        "B": {"ci95_low": 0.72, "ci95_high": 0.76},
    }
    separated = {
        "A": {"ci95_low": 0.80, "ci95_high": 0.84},
        "B": {"ci95_low": 0.70, "ci95_high": 0.75},
    }
    assert not significance(overlap, "A", "B")
    assert significance(separated, "A", "B")


def test_selective_reporting_truth_has_table_and_figure_scope():
    items = [e for r in _records() for e in r["errors"] if e["error_type"] == "selective_reporting"]
    assert items
    assert all(e["affected_artifact"] == "results.csv+visual.png" for e in items)
    assert all("reported subset" in e["affected_elements"] for e in items)


def test_agent_view_has_no_leakage():
    forbidden = ("truth", "reference", "canonical", "injector", "error_type", "clean", "corrupt", "template")
    for split in ("dev", "test_seen", "test_unseen"):
        for case in (DATA / "agent_view" / split).iterdir():
            text = "\n".join(
                p.read_text(errors="ignore") for p in case.iterdir() if p.suffix in {".json", ".csv", ".txt"}
            ).lower()
            assert not any(token in text for token in forbidden)


def test_ruleaudit_v2_cannot_access_evaluator():
    evaluator_case = next((DATA / "evaluator_view/dev").iterdir())
    with pytest.raises(V2AccessViolation):
        V2Reader(evaluator_case)
    agent_case = next((DATA / "agent_view/dev").iterdir())
    reader = V2Reader(agent_case)
    with pytest.raises(V2AccessViolation):
        reader.json("truth.json")


def test_v2_audit_results_read_only_allowed_files():
    allowed = {"protocol.json", "observations.csv", "results.csv", "visual.png", "statement.txt"}
    for result in (OUTPUT / "rule_audit").glob("*/*/audit_result.json"):
        payload = json.loads(result.read_text())
        assert set(payload["read_files"]) == allowed


def test_single_numeric_repair_reaudits_clean(tmp_path):
    target = next(
        r for r in _records()
        if len(r["errors"]) == 1 and r["errors"][0]["error_type"] == "numeric_copy_error"
    )
    source = DATA / target["paths"]["agent"]
    repaired = OUTPUT / "rule_audit" / target["split"] / target["case_id"] / "repaired_results.csv"
    case = tmp_path / "data_v2" / "agent_view" / target["split"] / target["case_id"]
    shutil.copytree(source, case)
    shutil.copy2(repaired, case / "results.csv")
    result = audit_case(case, tmp_path / "audit" / target["case_id"], ROOT)
    assert not result["predicted_errors"]
