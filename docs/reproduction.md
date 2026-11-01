# Results and reproduction / 结果与复算

[Home](../README.md) · [中文主页](../README.zh-CN.md)

## Three distinct operations

1. `demo`: run the original deterministic auditor on one known dev task. Gold is read after prediction, not supplied to the auditor.
2. `ruleaudit`: run original RuleAudit and its evaluator into a new directory under `runs/`. `--split all` covers the 120 main tasks.
3. `rescore`: evaluate exported saved labels using the original Direct evaluator's shared classification/localization aggregate function. No model is called; token usage, latency, costs and repair metrics cannot be recovered from this label export and are omitted.

演示、确定性重新运行、历史预测重新评分是不同操作。历史标签重评分不等于模型重跑，更不能恢复账单、时延、回答正文或完整provider来源链。

## Frozen manuscript identity

The R4 camera-ready SHA256 is `dda1a283595140321d88f7e28c690dcb4aadf52f001db9fbf43867aa9189996a`. The manuscript bytes in source commit `63750a49c7ea1528f34da3fbd3cfabc727f338db` match the separately held final artifact. PDF, manuscript source and conference templates are not redistributed here. This binding selects the manuscript version; it is not an external publication confirmation.

## Manuscript-to-public-release correspondence

| Manuscript item | Public material | Verification coverage |
|---|---|---|
| Table I: task composition | `data_v2/split_manifest.json`, evaluator task records, schemas | Main data validator checks all 120 tasks |
| Table II: RuleAudit, Direct, Tool | `historical_predictions/{ruleaudit,direct,tool}.jsonl`, evaluator truth, original `evaluate_direct_mllm.aggregate` | Saved-label classification/localization metrics recomputed; model execution not repeated |
| Table II: evidence-only control | Manuscript anchor: micro-F1 0.1728, Exact Set 0.2917 | Not reconstructed by this release; do not infer a reproduction from the other three rows |
| Table III: scale check | Separate 360-task data and `historical_predictions/ruleaudit_scale.jsonl` | Saved-label classification/localization metrics recomputed separately |
| Table IV: split diagnostics | Main per-task labels retain split identities | Underlying labels available; public rescore CLI reports pooled metrics |
| Table V: 60-task cross-model diagnostic | Selection manifests and reference configuration only | Cross-model historical predictions/provider evidence not exported; not reproduced |
| Table VI: per-error breakdown | Main labels and original aggregate function | Counts can be inspected from the original aggregate; public CLI reports pooled metrics |
| Table VII: execution reliability | Counts and caveats remain manuscript claims | Raw requests, repair-event histories and billing records excluded; not reconstructed |
| Table VIII: evidence ablation | Method implementation and prompts | Variant prediction sets not exported; not reproduced |
| HybridAudit diagnostic | `configs/hybrid_gate_dev_frozen.json`; manuscript reports no added gain | No new gate selection or public diagnostic rerun |

The correspondence deliberately does not imply complete reproduction of every paper table. No missing historical prediction is fabricated. See `manifests/paper_results.json` for machine-readable anchors and coverage.

## Main classification/localization metrics

| Method | n | Micro-F1 | Macro-F1 | Exact Set | Localization | Clean FPR |
|---|---:|---:|---:|---:|---:|---:|
| RuleAudit | 120 | 0.8708 | 0.8950 | 0.7917 | 0.8875 | 0.1000 |
| Direct | 120 | 0.3909 | 0.3681 | 0.4083 | 0.2271 | 0.2333 |
| Tool | 119 | 0.6068 | 0.5235 | 0.5294 | 0.3032 | 0.1667 |

Tool is missing `test_seen/v2_449a4b53b64285c4`. Default `exclude` scores 119 tasks; `count_as_miss` scores all 120, marks the missing prediction as a failure and yields micro-F1 0.6017. No empty success is invented for the missing task. The CLI lists the missing ID and both planned and available counts.

Tool 默认排除缺失预测，分母119；固定分母敏感性分析把缺失任务计为未命中，分母120，Micro-F1为0.6017。两者含义不同。

The separate scale check scores 360 saved RuleAudit predictions: micro-F1 0.8782, macro-F1 0.8977, Exact Set 0.7944, localization 0.9009 and Clean FPR 0.1222. Nine semantic/evidence unknowns were reported historically; this does not establish natural scientific validity.

The manuscript's 60-task cross-model diagnostic has backend-dependent gains, including a GPT-4.1 counterexample, and a 59/60 Direct denominator for that backend. The evidence ablation's 23/24, 22/24 and 23/24 scored populations differ. These diagnostic comparisons cannot establish causal effects on matched populations. They are not rerun in this release.

## Model-backed runs

`src/agents/`, `src/providers/`, original prompts and reference settings preserve the implementations. The historical runner entry points retain frozen-state checks. Some checks depend on provider outputs and manifests excluded from this public release; a fresh public checkout is therefore not a complete executable model-study reproduction. Do not disable those checks to imply original protocol compliance.

Running providers can incur charges and requires credentials, compatible service access, frozen inputs, logging and a separately defined execution scope. Model availability and provider behavior may have changed. Public setup, tests and demos never request providers.

模型组件用于代码审查。原历史运行器的冻结校验依赖本版未附带的材料，不能声称公开仓库支持一键重跑模型研究，也不应绕过校验后继续宣称符合原协议。

## Public release verification

At release preparation, the 120 and 360 data validators passed independently, and the original main benchmark freeze verified. A fresh 120-task RuleAudit run into a new `runs/` directory matched the six main reported classification/localization metrics; original deterministic outputs remained unchanged. This is a local software/recalculation check, not independent scientific confirmation. The precise checks and interpreter are recorded in `manifests/release_validation.json`.
