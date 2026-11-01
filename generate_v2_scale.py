from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from generate_v2 import (
    CHARTS,
    case_id,
    choose_models,
    clean_table,
    compatible_chart,
    compatible_experiment,
    inject,
    make_plot_spec,
    make_raw,
    write_csv,
    write_json,
)
from src.v2.claims import TEMPLATES, make_claim
from src.v2.core import ERROR_TYPES
from src.v2.plots import render_plot, style_for

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "data_v2_scale"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "v2_scale"


COMBO_PATTERNS = [
    ["numeric_copy_error", "ranking_error"],
    ["legend_swap", "relative_improvement_error"],
    ["aggregation_error", "significance_overclaim"],
    ["axis_scale_error", "selective_reporting"],
    ["metric_name_mismatch", "ablation_conclusion_error"],
    ["series_data_swap", "numeric_copy_error"],
    ["error_bar_error", "significance_overclaim"],
    ["relative_improvement_error", "metric_name_mismatch", "ranking_error"],
    ["selective_reporting", "significance_overclaim", "axis_scale_error"],
    ["legend_swap", "numeric_copy_error", "relative_improvement_error"],
]


def parse_split_sizes(text: str) -> dict[str, int]:
    result = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        result[name.strip()] = int(value.strip())
    if not result:
        raise ValueError("at least one split size is required")
    return result


def split_alias(split: str) -> str:
    lowered = split.lower()
    if "unseen" in lowered:
        return "test_unseen"
    if "seen" in lowered:
        return "test_seen"
    if "dev" in lowered:
        return "dev"
    return "dev"


def split_seed_bases(splits: dict[str, int], seed_base: int) -> dict[str, int]:
    return {split: seed_base + i * 1_000_000 for i, split in enumerate(splits)}


def allocate_counts(split_sizes: dict[str, int]) -> dict[str, tuple[int, int, int]]:
    total = sum(split_sizes.values())
    target_clean = round(total * 0.25)
    target_single = round(total * 0.50)

    def largest_remainder(target: int) -> dict[str, int]:
        exact = {split: split_sizes[split] * target / total for split in split_sizes}
        floors = {split: int(value) for split, value in exact.items()}
        remaining = target - sum(floors.values())
        order = sorted(split_sizes, key=lambda split: (exact[split] - floors[split], split_sizes[split]), reverse=True)
        for split in order[:remaining]:
            floors[split] += 1
        return floors

    clean_counts = largest_remainder(target_clean)
    single_counts = largest_remainder(target_single)
    result = {}
    for split, size in split_sizes.items():
        clean = clean_counts[split]
        single = single_counts[split]
        combo = size - clean - single
        if combo < 0:
            raise RuntimeError(f"negative combo allocation for {split}")
        result[split] = (clean, single, combo)
    return result


def category_plan(split_sizes: dict[str, int], seed_bases: dict[str, int]) -> dict[str, list[list[str]]]:
    counts = allocate_counts(split_sizes)
    singles = [ERROR_TYPES[i % len(ERROR_TYPES)] for i in range(sum(v[1] for v in counts.values()))]
    combos = [COMBO_PATTERNS[i % len(COMBO_PATTERNS)] for i in range(sum(v[2] for v in counts.values()))]
    result, si, ci = {}, 0, 0
    for split, (clean_n, single_n, combo_n) in counts.items():
        plan = [[] for _ in range(clean_n)]
        plan += [[error] for error in singles[si:si + single_n]]
        si += single_n
        plan += [list(errors) for errors in combos[ci:ci + combo_n]]
        ci += combo_n
        rng = np.random.default_rng(seed_bases[split] + 777)
        rng.shuffle(plan)
        result[split] = plan
    return result


def build_summary(records: list[dict]) -> dict:
    return {
        "task_count": len(records),
        "by_split": dict(Counter(r["split"] for r in records)),
        "by_error_count": dict(Counter(len(r["errors"]) for r in records)),
        "by_experiment_type": dict(Counter(r["experiment_type"] for r in records)),
        "by_chart_type": dict(Counter(r["chart_type"] for r in records)),
        "by_error_type": dict(Counter(e["error_type"] for r in records for e in r["errors"])),
    }


def generate(args: argparse.Namespace) -> dict:
    data_root = Path(args.output_data_root).resolve()
    output_root = Path(args.output_root).resolve()
    if data_root == ROOT / "data_v2" or output_root == ROOT / "outputs" / "v2":
        raise SystemExit("refusing to write scale benchmark into frozen V2 roots")
    split_sizes = parse_split_sizes(args.split_sizes)
    seed_bases = split_seed_bases(split_sizes, args.seed_base)
    plans = category_plan(split_sizes, seed_bases)
    dry_records = []
    for split, plan in plans.items():
        for local_index, errors in enumerate(plan):
            seed = seed_bases[split] + local_index * 997 + 13
            dry_records.append({"split": split, "seed": seed, "errors": [{"error_type": e} for e in errors]})
    if args.dry_run:
        return {
            "dry_run": True,
            "data_root": str(data_root),
            "output_root": str(output_root),
            "version": args.version,
            "split_sizes": split_sizes,
            "seed_bases": seed_bases,
            "planned_summary": build_summary([
                {"split": r["split"], "errors": r["errors"], "experiment_type": "planned", "chart_type": "planned"}
                for r in dry_records
            ]),
        }

    for path in (data_root / "agent_view", data_root / "evaluator_view", output_root):
        path.mkdir(parents=True, exist_ok=True)
    split_manifest = {"version": args.version, "splits": {}, "isolation": {}}
    all_records = []
    global_index = 0
    for split, plan in plans.items():
        alias = split_alias(split)
        refs = []
        for local_index, errors in enumerate(plan):
            seed = seed_bases[split] + local_index * 997 + 13
            rng = np.random.default_rng(seed)
            experiment_type = compatible_experiment(errors, global_index)
            model_count = 4 if experiment_type == "ablation" else int(rng.integers(2, 6))
            models = choose_models(rng, model_count, experiment_type)
            raw, summary, metric = make_raw(seed, experiment_type, models)
            chart = compatible_chart(errors, global_index, experiment_type)
            if chart not in CHARTS:
                raise RuntimeError(f"unexpected chart type: {chart}")
            style = style_for(alias, seed)
            template_index = local_index % len(TEMPLATES[alias])
            table = clean_table(summary, experiment_type, metric)
            claim = make_claim(summary, metric, alias, template_index, experiment_type)
            plot_spec = make_plot_spec(summary, raw, experiment_type, metric, chart, style)
            observed_table, observed_claim, observed_spec, truth = inject(
                errors, table, claim, plot_spec, summary, metric, experiment_type, rng
            )
            cid = case_id(split, seed)
            agent = data_root / "agent_view" / split / cid
            evaluator = data_root / "evaluator_view" / split / cid
            agent.mkdir(parents=True, exist_ok=True)
            evaluator.mkdir(parents=True, exist_ok=True)
            write_csv(agent / "observations.csv", raw)
            protocol = {
                "case_id": cid,
                "experiment_type": experiment_type,
                "primary_metric": metric,
                "metric_directions": {metric: "minimize" if metric in {"MAE", "RMSE"} else "maximize"},
                "model_count": len(models),
                "chart_type": chart,
                "render_config": style,
                "files": {"raw": "observations.csv", "table": "results.csv", "figure": "visual.png", "claim": "statement.txt"},
            }
            write_json(agent / "protocol.json", protocol)
            write_csv(agent / "results.csv", observed_table)
            (agent / "statement.txt").write_text(observed_claim)
            render_plot(observed_spec, agent / "visual.png")

            write_csv(evaluator / "observations.csv", raw)
            write_json(evaluator / "protocol.json", protocol)
            write_csv(evaluator / "reference_table.csv", table)
            (evaluator / "reference_statement.txt").write_text(claim)
            render_plot(plot_spec, evaluator / "reference_visual.png")
            shutil.copy2(agent / "results.csv", evaluator / "observed_table.csv")
            shutil.copy2(agent / "statement.txt", evaluator / "observed_statement.txt")
            shutil.copy2(agent / "visual.png", evaluator / "observed_visual.png")
            write_json(evaluator / "canonical_facts.json", {
                "summary": summary,
                "primary_metric": metric,
                "models": models,
                "plot_spec": plot_spec,
                "observed_plot_spec": observed_spec,
            })
            write_json(evaluator / "truth.json", {"case_id": cid, "split": split, "errors": truth})
            record = {
                "case_id": cid,
                "split": split,
                "experiment_type": experiment_type,
                "chart_type": chart,
                "seed": seed,
                "errors": truth,
                "paths": {"agent": f"agent_view/{split}/{cid}", "evaluator": f"evaluator_view/{split}/{cid}"},
                "versions": {"generator": args.version, "injector": args.version},
                "template_fingerprint": hashlib.sha256(TEMPLATES[alias][template_index].encode()).hexdigest()[:12],
                "style_fingerprint": hashlib.sha256(json.dumps(style, sort_keys=True).encode()).hexdigest()[:12],
                "template_family": alias,
            }
            write_json(evaluator / "task.json", record)
            refs.append(cid)
            all_records.append(record)
            global_index += 1
        split_manifest["splits"][split] = refs
    split_manifest["isolation"] = {
        "seed_ranges": {
            split: [
                min(r["seed"] for r in all_records if r["split"] == split),
                max(r["seed"] for r in all_records if r["split"] == split),
            ]
            for split in split_sizes
        },
        "template_fingerprints": {
            split: sorted({r["template_fingerprint"] for r in all_records if r["split"] == split})
            for split in split_sizes
        },
        "style_fingerprints": {
            split: sorted({r["style_fingerprint"] for r in all_records if r["split"] == split})
            for split in split_sizes
        },
        "template_families": {split: split_alias(split) for split in split_sizes},
    }
    write_json(data_root / "split_manifest.json", split_manifest)
    summary = build_summary(all_records)
    summary["version"] = args.version
    summary["data_root"] = str(data_root)
    write_json(output_root / "benchmark_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a non-frozen ArtifactAuditBench V2 scale benchmark.")
    parser.add_argument("--output-data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split-sizes", default="scale_dev=90,scale_test_seen=135,scale_test_unseen=135")
    parser.add_argument("--seed-base", type=int, default=510_000)
    parser.add_argument("--version", default="2.0.0-scale")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
