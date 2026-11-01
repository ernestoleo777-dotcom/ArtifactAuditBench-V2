# Source provenance and public boundary / 来源与公开范围

[Home](../README.md) · [中文主页](../README.zh-CN.md)

## Source selection

Public assets were extracted using an explicit allowlist from source commit `63750a49c7ea1528f34da3fbd3cfabc727f338db`. The source is a recovered mixed research archive. Its Git commit binds the saved snapshot; it does not prove original development chronology or complete pre-recovery history.

The R4 manuscript PDF stored at that commit matches final camera-ready SHA256 `dda1a283595140321d88f7e28c690dcb4aadf52f001db9fbf43867aa9189996a`. Older root READMEs and earlier submitted PDFs are not substituted for this binding.

`manifests/source_inventory.json` distinguishes byte-exact files from transformed label exports and one public path normalization. In `data_v2/model_sweep_subset_60.json`, the original machine-local source root was replaced with relative `data_v2`; task membership is unchanged. Main frozen benchmark files were not changed.

## Structured label export

`historical_predictions/` contains only task/split identity, prediction status, error types, affected-element identifiers, original result path and SHA256. It omits model prose, suggestions, tokens, costs, provider metadata, caches and requests. Each exported row can be compared to the original private record using its source hash; the public release alone does not authenticate missing provider transport evidence.

There are 120 RuleAudit, 120 Direct and 119 Tool main predictions, plus 360 separate RuleAudit scale-check predictions. Missing records remain missing. No generated label replaces the absent Tool result.

## Included rights classes

The code is author-controlled implementation; datasets consist of synthetic observations, templated claims and charts produced by project code. Reference names are synthetic model labels, not copied scientific papers or personal data. External Python dependencies are referenced rather than vendored. The release explicitly states separate scope for code, synthetic data and structured annotations; it does not grant rights to excluded provider responses or papers.

## Excluded classes

Private environment files, credentials, private machine paths, raw responses, API request logs/caches, model weights, papers/PDFs, submission-system exports, reviewer correspondence, copyright forms, conference templates, backups, environments, later SAI/repair/benchmark extensions, internal decisions and unrelated projects are excluded. The public manifest does not enumerate confidential excluded filenames.

## Release identity

`manifests/public_files.json` binds all public file bytes except itself to avoid recursive hashing. Git commit and release tag bind the manifest. New CLI, public checks, tests, licensing text and documentation are public-release additions, not original scientific execution artifacts.

源材料只读提取，历史档案未迁移、删除或重写。新增公共接口与文档的身份和原始实验代码分开记录；公开版本号不改变V2的科学版本。
