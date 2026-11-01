# ArtifactAuditBench V2

**English** | [中文说明](README.zh-CN.md)

**ArtifactAuditBench V2: A Multimodal Benchmark for Auditing Scientific Experiment Artifacts**

Hongchen Li · Institute for Advanced Study, Shenzhen University

A controlled benchmark for checking consistency among machine-readable experimental observations, reported tables, rendered figures, and scientific claims. The main benchmark contains **120 synthetic tasks**, **12 error families**, and **7 chart families**. A separate **360-task synthetic scale check** is also included.

[Quick start](#quick-start) · [Benchmark and data](docs/benchmark.md) · [Results and reproduction](docs/reproduction.md) · [Limits](docs/limitations.md) · [Citation](CITATION.cff)

## What this release provides

- Frozen main and scale data, with separate agent-visible and evaluator-only views.
- Original V2 generation, RuleAudit, Direct MLLM and Tool-agent implementation components, prompts and schemas extracted from the R4 custody commit.
- A public offline command interface, known-dev demonstration and fresh-output RuleAudit execution.
- Saved **structured prediction labels** for recomputing the main comparison and scale-check metrics without model calls. Provider prose, responses, caches and request logs are excluded.
- Explicit file hashes, source mapping and offline checks.

This is a public software release of V2, not a new benchmark experiment or V3. The source commit also contains later research; those extensions are excluded here. The R4 manuscript hash matches the separately held final camera-ready artifact. Manuscript identity does not by itself verify every historical experiment.

## Quick start

Run from the repository root. CI targets Python 3.11; local release checks also used Python 3.13.13. Other interpreter versions are not qualified here. The direct dependency pins are in `requirements.lock.txt`; transitive packages are resolved by pip. Installation accesses package infrastructure. The following benchmark commands run locally without API keys, models or further downloads.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m artifactauditbench demo
```

Windows activation: `.venv\Scripts\activate`.

The demo audits one explicitly known dev case, then compares its predicted error labels with evaluator truth. It is not a held-out result or a new scientific finding.

```bash
# Validate the 120 main tasks and the separate 360-task scale check
python -m artifactauditbench validate
python -m artifactauditbench validate --scale

# Run RuleAudit on dev; use a new runs/ subdirectory for each invocation
python -m artifactauditbench ruleaudit --split dev --output runs/my-dev-check

# Recompute saved-label metrics without rerunning either model baseline
python -m artifactauditbench rescore --method ruleaudit
python -m artifactauditbench rescore --method direct
python -m artifactauditbench rescore --method tool
python -m artifactauditbench rescore --method tool --failure-policy count_as_miss
python -m artifactauditbench rescore --method ruleaudit_scale
```

RuleAudit supports `--split all`. It reads the public task view; evaluation subsequently reads ground truth. Public CLI outputs are restricted to a fresh directory under `runs/`. See [reproduction details](docs/reproduction.md) before invoking historical scripts directly.

## Main manuscript anchors

| Method | Scored / planned | Micro-F1 | Exact error set |
|---|---:|---:|---:|
| RuleAudit | 120/120 | 0.8708 | 0.7917 |
| Direct MLLM | 120/120 | 0.3909 | 0.4083 |
| Tool-augmented Single-pass Agent | 119/120 | 0.6068 | 0.5294 |

The public saved labels reproduce these classification/localization metrics. Tool excludes one missing prediction in the original scored-population comparison. Counting it as a miss gives micro-F1 0.6017 on 120 tasks. The separate 360-task RuleAudit scale check gives micro-F1 0.8782; do not pool the datasets. Full correspondence and unreproduced diagnostics are documented in [reproduction.md](docs/reproduction.md).

RuleAudit uses raw experimental artifacts and known claim/rendering representations. This controlled advantage is part of its scope. HybridAudit showed no additional gain under the manuscript's gate; natural transfer, a general LLM advantage, and LLM necessity are not established.

## Repository map

| Path | Purpose |
|---|---|
| `artifactauditbench/` | New offline public CLI |
| `src/v2/`, `src/tools/` | Original deterministic baseline and evidence tools |
| `src/agents/`, `src/providers/` | Original model baseline components; provider execution requires excluded provenance assets |
| `data_v2/` | Main data: 30 dev / 45 seen-test / 45 unseen-test |
| `data_v2_scale/` | Separate 360-task scale-check data |
| `historical_predictions/` | Structured label exports with source-record hashes |
| `outputs/v2/rule_audit/` | Frozen original deterministic outputs and repair artifacts |
| `configs/` | Schemas, prompts, model-reference settings and gate configuration |
| `manifests/` | Source mapping, paper identity, result correspondence and file inventory |
| `tests/`, `.github/workflows/` | Offline software/data checks |

## Publication, citation and licensing

Local records report acceptance as ICTAI 2026 Full Paper #175, followed by a withdrawal request pending official confirmation. Official publication and withdrawal disposition have not been independently verified for this release. This repository is not evidence of publication. See [publication status](docs/publication_status.md); no DOI or proceedings claim is invented.

Use [CITATION.cff](CITATION.cff) for the software and manuscript identity. [Apache-2.0](LICENSE) covers author-owned code and original repository documentation. The separately scoped [data license](DATA_LICENSE.md) explicitly covers the included author-generated synthetic assets and structured annotations. It does not extend to excluded provider responses, manuscripts, model weights, conference templates or external dependencies. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Verification and limits

```bash
python scripts/check_release.py
python freeze_v2_benchmark.py
python -m pytest -q tests
```

CI validates data and hashes, runs core tests, the dev demonstration, saved-label rescoring and a fresh dev RuleAudit run. Checks establish bounded software behavior and artifact identity, not independent scientific validation. All test labels are public: the release is for inspection and reproducible evaluation, not an assurance of future evaluation blindness.

[Source provenance](docs/provenance.md) · [Limitations](docs/limitations.md) · [Changelog](CHANGELOG.md)
