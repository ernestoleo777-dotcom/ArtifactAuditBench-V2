# ArtifactAuditBench V2

[English](README.md) | **中文说明**

**ArtifactAuditBench V2: A Multimodal Benchmark for Auditing Scientific Experiment Artifacts**

Hongchen Li · 深圳大学高等研究院

用于检查科学实验的原始观测、报告表格、程序绘制图像与文字主张是否一致的受控基准。主基准包含 **120 个合成任务、12 类错误、7 类图形**，另附 **360 任务合成规模检查**，两者独立统计。

[数据与任务](docs/benchmark.md) · [复算与结果对应](docs/reproduction.md) · [限制](docs/limitations.md) · [引用](CITATION.cff)

## 本版包含什么

- 主基准及规模检查的冻结数据，分离模型可见视图和评价器真值。
- 从 R4 保存提交提取的 V2 生成器、RuleAudit、Direct MLLM、Tool-agent 相关实现、提示词和 Schema。
- 新增的离线命令入口、已知开发集演示，以及写入新目录的 RuleAudit 运行流程。
- 可以离线重算主要指标的历史结构化预测标签；不包含模型回答正文、provider 原始响应、缓存或请求日志。
- 公开文件清单、来源对应关系、哈希及自动检查。

这是 V2 的公开软件版本，不是 V3 或新增研究结果。源提交还包含后续研究，本仓库没有一并导出。R4 稿件哈希已与另存的最终 camera-ready 核对一致；这不等于独立验证了全部历史实验。

## 快速开始

在仓库根目录运行。CI使用Python 3.11，本地发布检查另使用Python 3.13.13；其他解释器版本尚未验证。`requirements.lock.txt` 固定直接依赖版本，间接依赖仍由 pip 解析。安装依赖需要网络；以下基准命令无需密钥、模型权重或额外数据下载。

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m artifactauditbench demo
```

Windows 使用 `.venv\Scripts\activate` 激活环境。

演示先让 RuleAudit 读取一个明确标注的已知 dev 案例，再读取评价真值对比错误标签。这不是未见测试，也不产生新的科学结论。

```bash
# 分别校验120任务主基准与360任务规模检查
python -m artifactauditbench validate
python -m artifactauditbench validate --scale

# 运行dev基线并评分；每次选择新的 runs/ 子目录
python -m artifactauditbench ruleaudit --split dev --output runs/my-dev-check

# 重算保存标签的指标，不重新调用模型
python -m artifactauditbench rescore --method ruleaudit
python -m artifactauditbench rescore --method direct
python -m artifactauditbench rescore --method tool
python -m artifactauditbench rescore --method tool --failure-policy count_as_miss
python -m artifactauditbench rescore --method ruleaudit_scale
```

RuleAudit 支持 `--split all`。审计阶段读取公开任务视图，后续评价阶段才读取真值。统一入口仅向全新的 `runs/` 子目录写入。直接执行历史脚本前，请先阅读[复现说明](docs/reproduction.md)，以免覆盖保存材料。

## 原稿主要结果

| 方法 | 已计分 / 计划任务 | Micro-F1 | 错误集合完全匹配 |
|---|---:|---:|---:|
| RuleAudit | 120/120 | 0.8708 | 0.7917 |
| Direct MLLM | 120/120 | 0.3909 | 0.4083 |
| Tool-augmented Single-pass Agent | 119/120 | 0.6068 | 0.5294 |

本仓库保存的结构化标签可以重算这些分类及定位指标。Tool 原对照排除一个缺失预测；将该任务按未命中计入120分母，Micro-F1 为0.6017。独立360任务规模检查的 RuleAudit Micro-F1 为0.8782，不能与120任务合并。其他诊断的复算覆盖与缺口见[结果对应说明](docs/reproduction.md)。

RuleAudit 能访问原始产物，并利用已知主张与渲染表示。这种受控优势需要保留。原稿 HybridAudit 在对应门控下没有新增收益；自然错误迁移、通用模型优势及 LLM 必要性均未建立。

## 目录导航

| 路径 | 内容 |
|---|---|
| `artifactauditbench/` | 新增离线公共入口 |
| `src/v2/`、`src/tools/` | 原始确定性基线及证据工具 |
| `src/agents/`、`src/providers/` | 原始模型方法组件；运行仍涉及本版未公开的来源验证材料 |
| `data_v2/` | 30 dev、45 seen-test、45 unseen-test 主数据 |
| `data_v2_scale/` | 独立360任务规模检查 |
| `historical_predictions/` | 结构化历史预测及源记录哈希 |
| `outputs/v2/rule_audit/` | 冻结的历史确定性输出和修复产物 |
| `configs/` | Schema、提示词、模型参考设置和门控配置 |
| `manifests/` | 来源、稿件身份、结果对应和文件清单 |
| `tests/`、`.github/workflows/` | 本地测试与自动检查 |

## 论文状态、引用与许可

本地记录为 ICTAI 2026 Full Paper #175 已接收，之后已请求撤稿、等待正式确认。本轮未独立确认最终出版或撤稿状态；GitHub 公开不代表论文正式发表。详见[状态说明](docs/publication_status.md)。

[CITATION.cff](CITATION.cff) 提供软件与稿件引用信息。[Apache-2.0](LICENSE) 覆盖作者拥有权利的源码和原创仓库文档。[数据许可](DATA_LICENSE.md) 另行明确覆盖本版作者生成的合成资产和结构化标注；不向未包含的 provider 回答、论文、模型权重、会议模板或外部依赖授予许可。详见[第三方说明](THIRD_PARTY_NOTICES.md)。

## 检查与边界

```bash
python scripts/check_release.py
python freeze_v2_benchmark.py
python -m pytest -q tests
```

GitHub 自动校验数据与哈希、运行核心测试、dev演示、历史标签复算和新的dev RuleAudit运行。通过代表限定的软件行为与文件身份检查，不代表独立科学验证。全部测试标签均已公开，不能保证后续使用者对这些案例保持未见状态。

[来源与公开范围](docs/provenance.md) · [完整限制](docs/limitations.md) · [更新记录](CHANGELOG.md)
