# Changelog

## 2026-11-16 — Paper status correction

- Record the author-reconfirmed ICTAI 2026 Full Paper acceptance (#175) and completed camera-ready consistently in both READMEs, citation metadata and paper identity metadata.
- Supersede the initial release's outdated publication-status wording. Do not assign an unverified DOI or proceedings publication date.
- Refresh the public file inventory; preserve scientific code, datasets, saved predictions and reported results.

## 2.0.0-public.1 — 2026-11-02

- Create an explicitly scoped public V2 release from the R4 custody commit; preserve original scientific code and frozen main/scale data.
- Add English/Chinese READMEs, benchmark format, publication status, reproduction coverage and limitations.
- Add offline dev demonstration, data validation, fresh-output RuleAudit execution and saved-label rescoring CLI.
- Export structured labels only: RuleAudit 120, Direct 120, Tool 119, separate scale RuleAudit 360. Preserve the missing Tool task and alternative failure denominator.
- Add file/source manifests, scoped code/data licensing, offline tests and GitHub checks.
- Verify main and scale data, saved-label anchors and a fresh 120-task RuleAudit run. No new provider inference, training, dataset generation or V3 scientific claim.
