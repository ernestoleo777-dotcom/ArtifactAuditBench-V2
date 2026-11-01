# Benchmark and data / 基准与数据

[Home](../README.md) · [中文主页](../README.zh-CN.md)

## Main population

The main population has 120 controlled synthetic tasks: 30 dev, 45 test-seen and 45 test-unseen. There are 30 clean tasks, 60 single-error tasks and 30 combination tasks (21 double-error, 9 triple-error), with 129 error instances. Experiment types cover repeated runs (48), ablation (23), multiclass classification (22), binary classification (18) and regression (9).

主数据为120个受控合成任务，包含30个干净任务、60个单错误任务、30个组合错误任务，共129个错误实例。它不是自然论文错误的抽样。

The 360-task scale check is separately generated and split 90/135/135. It contains 387 error instances. It checks synthetic scale stability, not natural transfer, and is never pooled with the main study.

360任务规模检查单独生成、单独统计，划分90/135/135，共387个错误实例；不能把它与主研究合并为480个独立主任务。

## Agent-visible data

Each `agent_view/<split>/<task_id>/` directory contains exactly:

| File | Meaning |
|---|---|
| `observations.csv` | Machine-readable synthetic raw observations / 原始观测 |
| `results.csv` | Reported results / 报告表格 |
| `visual.png` | Rendered chart / 程序绘制图形 |
| `statement.txt` | Claim to audit / 待审计主张 |
| `protocol.json` | Public measurement and rendering context / 公开实验与绘图上下文 |

RuleAudit recomputes from the raw observations and compares supported claims and rendered candidates. Its access and renderer/template knowledge are broader than a figure-only auditor's capabilities. A numeric discrepancy is not an independent human judgment of scientific validity.

## Evaluator view

`evaluator_view/` contains truth, reference artifacts, canonical facts and task metadata. These files are used after prediction for scoring. Do not feed them into model prompts or evidence packs. Schemas are in `configs/v2/`; split membership is in each dataset's `split_manifest.json`.

`evaluator_view` 含真值和参考产物，仅用于评价。模型输入应从 `agent_view` 构建，不能混入真值、参考图表或错误标签。发布真值以后，程序分离仍然必要，但不代表使用者对测试内容未见。

## Error taxonomy

- `numeric_copy_error`: numeric transcription / 数值抄录
- `relative_improvement_error`: relative-improvement claim / 相对提升计算
- `ranking_error`: ordering / 排序
- `legend_swap`: legend mapping / 图例映射
- `metric_name_mismatch`: metric identity / 指标名称
- `axis_scale_error`: axis scale / 坐标尺度
- `error_bar_error`: uncertainty display / 误差条
- `series_data_swap`: series assignment / 数据系列对应
- `aggregation_error`: aggregation / 汇总
- `significance_overclaim`: significance wording under the encoded rule / 编码规则下的显著性过度主张
- `ablation_conclusion_error`: ablation conclusion / 消融结论
- `selective_reporting`: reporting subset / 选择性报告

Chart families: bar, grouped bar, line, scatter, error bar, confusion heatmap and multi-panel. Error types are controlled generator labels, not an exhaustive taxonomy of research misconduct or scientific mistakes.

## Generators

Original `generate_v2.py` and `generate_v2_scale.py` are included for inspection and provenance. Their historical defaults target benchmark directories and may replace generated content. Do not execute them inside a frozen release checkout. For generation research, use a separate disposable copy and retain the original release hashes. The supported public quick start validates the shipped data instead of regenerating it. Rendering bytes can depend on fonts and library versions.

原始生成器用于核查构造过程，默认路径可能覆盖数据。请勿在冻结发布目录中直接运行；如需研究新生成数据，使用独立可丢弃副本并保留原哈希。新生成文件不自动成为历史论文输入。
