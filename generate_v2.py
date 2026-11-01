from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from src.generators.synthetic import generate_binary_data, generate_regression_data
from src.metrics import compute_metrics
from src.v2.claims import TEMPLATES, claim_context, make_claim
from src.v2.core import ERROR_TYPES, NEUTRAL_NAMES, multiclass_metrics, run_statistics
from src.v2.plots import render_plot, style_for

ROOT = Path(__file__).resolve().parent
VERSION = "2.0.0"
CHARTS = ["bar", "grouped_bar", "line", "scatter", "error_bar", "confusion_heatmap", "multi_panel"]
SPLIT_SIZES = {"dev": 30, "test_seen": 45, "test_unseen": 45}
SPLIT_SEED_BASE = {"dev": 110_000, "test_seen": 220_000, "test_unseen": 330_000}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_id(split: str, seed: int) -> str:
    return "v2_" + hashlib.sha256(f"{split}:{seed}:ArtifactAuditBench".encode()).hexdigest()[:16]


def choose_models(rng, count: int, experiment_type: str) -> list[str]:
    if experiment_type == "ablation":
        return ["Baseline", "Module A", "Module B", "Full Model"][:count]
    return list(rng.choice(NEUTRAL_NAMES, size=count, replace=False))


def make_raw(seed: int, experiment_type: str, models: list[str]) -> tuple[list[dict], dict, str]:
    rng = np.random.default_rng(seed)
    if experiment_type == "binary_classification":
        base = generate_binary_data(seed, 180)
        rows = [{"sample_id": r["sample_id"], "y_true": r["y_true"]} for r in base]
        for i, model in enumerate(models):
            latent = np.asarray([r["model_b_probability"] for r in base])
            probability = np.clip(latent + rng.normal(0, 0.08 + i * 0.018, len(base)) - i * 0.012, 0.001, 0.999)
            for j, row in enumerate(rows):
                row[f"{model}__probability"] = float(probability[j])
                row[f"{model}__prediction"] = int(probability[j] >= 0.5)
        converted = []
        for model in models:
            local = [
                {"y_true": r["y_true"], "model_a_prediction": r[f"{model}__prediction"],
                 "model_a_probability": r[f"{model}__probability"],
                 "model_b_prediction": r[f"{model}__prediction"], "model_b_probability": r[f"{model}__probability"]}
                for r in rows
            ]
            converted.append((model, compute_metrics(local, "binary_classification")["Model A"]))
        summary = dict(converted)
        return rows, summary, "Macro-F1"
    if experiment_type == "regression":
        base = generate_regression_data(seed, 180)
        rows = [{"sample_id": r["sample_id"], "y_true": r["y_true"]} for r in base]
        y = np.asarray([r["y_true"] for r in base])
        for i, model in enumerate(models):
            prediction = y + rng.normal(0.02 * i, 0.48 + i * 0.15, len(y))
            for j, row in enumerate(rows):
                row[f"{model}__prediction"] = float(prediction[j])
        summary = {}
        for model in models:
            local = [{"y_true": r["y_true"], "model_a_prediction": r[f"{model}__prediction"],
                      "model_b_prediction": r[f"{model}__prediction"]} for r in rows]
            summary[model] = compute_metrics(local, "regression")["Model A"]
        return rows, summary, "RMSE"
    if experiment_type == "multiclass":
        classes, n = 4, 220
        y = rng.integers(0, classes, n)
        rows = [{"sample_id": i, "y_true": int(y[i])} for i in range(n)]
        summary = {}
        for i, model in enumerate(models):
            correctness = max(0.48, 0.78 - i * 0.055)
            pred = y.copy()
            flip = rng.random(n) > correctness
            pred[flip] = (pred[flip] + rng.integers(1, classes, flip.sum())) % classes
            for j, row in enumerate(rows):
                row[f"{model}__prediction"] = int(pred[j])
            summary[model] = multiclass_metrics(y, pred, classes)
        return rows, summary, "Macro-F1"
    run_count = int(rng.integers(6, 11))
    rows, summary = [], {}
    base = 0.68 if experiment_type == "repeated_runs" else 0.61
    for i, model in enumerate(models):
        if experiment_type == "ablation":
            offsets = {"Baseline": 0.0, "Module A": 0.035, "Module B": 0.012, "Full Model": 0.055}
            center = base + offsets[model]
        else:
            center = base + (len(models) - i) * 0.018
        values = np.clip(rng.normal(center, 0.025 + i * 0.004, run_count), 0, 1)
        for run, value in enumerate(values):
            rows.append({"model": model, "run": run, "value": float(value)})
        stats = run_statistics(values.tolist())
        summary[model] = {"Macro-F1": stats["mean"], **stats, "runs": values.tolist()}
    return rows, summary, "Macro-F1"


def clean_table(summary: dict, experiment_type: str, metric: str) -> list[dict]:
    rows = []
    for model, values in summary.items():
        if experiment_type == "multiclass":
            row = {"model": model}
            for name in ("Accuracy", "Macro-F1", "Micro-F1", "Weighted-F1"):
                row[name] = f"{values[name]:.6f}"
            for name, value in values["per_class_recall"].items():
                row[f"Recall:{name}"] = f"{value:.6f}"
        elif experiment_type in {"repeated_runs", "ablation"}:
            row = {
                "model": model, "mean": f"{values['mean']:.6f}", "std": f"{values['std']:.6f}",
                "se": f"{values['se']:.6f}", "ci95_low": f"{values['ci95_low']:.6f}",
                "ci95_high": f"{values['ci95_high']:.6f}",
            }
        else:
            row = {"model": model, **{name: f"{value:.6f}" for name, value in values.items()}}
        rows.append(row)
    return rows


def make_plot_spec(summary: dict, raw: list[dict], experiment_type: str, metric: str, chart_type: str, style: dict) -> dict:
    labels = list(summary)
    values = [summary[m][metric] for m in labels]
    spec = {
        "chart_type": chart_type, "labels": labels, "values": values, "metric": metric,
        "style": style, "title": f"{metric} comparison", "axis_limits": None,
    }
    if chart_type == "error_bar":
        spec["errors"] = [summary[m].get("se", 0.0) for m in labels]
    if chart_type == "line":
        spec["series_x"] = [f"Run {i + 1}" for i in range(max(len(summary[m].get("runs", values)) for m in labels))]
        spec["series"] = [
            summary[m].get("runs", [summary[m][metric]] * len(spec["series_x"])) for m in labels
        ]
    if chart_type == "scatter":
        spec["scatter_x"] = [list(range(len(summary[m].get("runs", [0, 1, 2])))) for m in labels]
        spec["scatter_y"] = [summary[m].get("runs", [summary[m][metric]] * 3) for m in labels]
    if chart_type == "confusion_heatmap":
        first = labels[0]
        spec["matrix"] = summary[first].get("confusion_matrix", [[1, 0], [0, 1]])
    return spec


def replace_first_number(text: str, old: float, new: float) -> str:
    candidates = [f"{old:.2f}", f"{old:.3f}", f"{old:.4f}"]
    for value in candidates:
        if value in text:
            return text.replace(value, value.replace(value, f"{new:.{len(value.split('.')[1])}f}"), 1)
    return text


def inject(errors: list[str], table: list[dict], claim: str, spec: dict, summary: dict, metric: str, experiment_type: str, rng) -> tuple[list[dict], str, dict, list[dict]]:
    observed_table = [dict(row) for row in table]
    observed_claim = claim
    observed_spec = json.loads(json.dumps(spec))
    truth = []
    context = claim_context(summary, metric)
    for error in errors:
        if error == "numeric_copy_error":
            row = observed_table[int(rng.integers(0, len(observed_table)))]
            columns = [c for c in row if c != "model" and not c.startswith("Recall:")]
            column = str(rng.choice(columns))
            before = float(row[column])
            after = before + max(abs(before) * 0.12, 0.03)
            if column not in {"RMSE", "MAE", "std", "se", "ci95_low", "ci95_high"}:
                after = min(after, 0.999)
            row[column] = f"{after:.6f}"
            artifact, elements = "results.csv", [row["model"], column]
        elif error == "relative_improvement_error":
            before = context["rel"]
            after = before + 7.5
            observed_claim = replace_first_number(observed_claim, before, after)
            if observed_claim == claim:
                observed_claim += f" Relative improvement: {after:.2f}%."
            artifact, elements = "statement.txt", ["relative_improvement_percent"]
        elif error == "ranking_error":
            before, after = context["best"], context["other"]
            observed_claim = observed_claim.replace(before, after, 1)
            artifact, elements = "statement.txt", ["best model"]
        elif error == "legend_swap":
            before = list(observed_spec["labels"])
            after = list(before)
            after[0], after[1] = after[1], after[0]
            observed_spec["labels"] = after
            artifact, elements = "visual.png", before[:2]
        elif error == "metric_name_mismatch":
            before = metric
            choices = [m for m in ["Macro-F1", "Micro-F1", "Weighted-F1", "Accuracy"] if m != metric]
            after = str(rng.choice(choices))
            observed_claim = observed_claim.replace(metric, after)
            artifact, elements = "statement.txt", ["metric name"]
        elif error == "axis_scale_error":
            before = observed_spec.get("axis_limits")
            vals = observed_spec["values"]
            low, high = min(vals), max(vals)
            after = [max(0.0, low - (high - low) * 0.08), high + (high - low) * 0.08]
            observed_spec["axis_limits"] = after
            artifact, elements = "visual.png", ["axis limits"]
        elif error == "error_bar_error":
            before = list(observed_spec.get("errors", [summary[m].get("se", 0) for m in observed_spec["labels"]]))
            after = list(reversed(before))
            if after == before:
                after = [v * 1.8 + 0.002 for v in before]
            observed_spec["errors"] = after
            artifact, elements = "visual.png", ["error bars"]
        elif error == "series_data_swap":
            before = list(observed_spec["values"])
            after = list(before)
            after[0], after[1] = after[1], after[0]
            observed_spec["values"] = after
            if "series" in observed_spec:
                observed_spec["series"][0], observed_spec["series"][1] = observed_spec["series"][1], observed_spec["series"][0]
            artifact, elements = "visual.png", observed_spec["labels"][:2]
        elif error == "aggregation_error":
            row = observed_table[0]
            before = float(row["mean"])
            after = summary[row["model"]]["max"]
            row["mean"] = f"{after:.6f}"
            artifact, elements = "results.csv", [row["model"], "mean"]
        elif error == "significance_overclaim":
            before = "no statistically reliable gain"
            after = "a statistically significant improvement"
            if before in observed_claim:
                observed_claim = observed_claim.replace(before, after)
            else:
                observed_claim += " The improvement is statistically significant."
            artifact, elements = "statement.txt", ["significance"]
        elif error == "ablation_conclusion_error":
            before = "Full Model"
            after = "Module B"
            observed_claim = observed_claim.replace(before, after, 1)
            observed_claim = observed_claim.replace("full model", "module B")
            artifact, elements = "statement.txt", ["ablation conclusion"]
        elif error == "selective_reporting":
            row = observed_table[0]
            before = float(row["mean"])
            after = summary[row["model"]]["max"]
            row["mean"] = f"{after:.6f}"
            observed_spec["values"][0] = after
            artifact, elements = "results.csv+visual.png", [row["model"], "reported subset"]
        else:
            raise ValueError(error)
        truth.append({
            "error_type": error, "affected_artifact": artifact, "affected_elements": elements,
            "before": before, "after": after, "repair": before,
        })
    return observed_table, observed_claim, observed_spec, truth


def compatible_experiment(errors: list[str], index: int) -> str:
    if "ablation_conclusion_error" in errors:
        return "ablation"
    if any(e in errors for e in ["error_bar_error", "aggregation_error", "significance_overclaim", "selective_reporting"]):
        return "repeated_runs"
    if "metric_name_mismatch" in errors:
        return "multiclass"
    return ["binary_classification", "regression", "multiclass", "repeated_runs", "ablation"][index % 5]


def compatible_chart(errors: list[str], index: int, experiment_type: str) -> str:
    if "error_bar_error" in errors:
        return "error_bar"
    if experiment_type == "multiclass" and index % 3 == 0 and not any(e in errors for e in ["legend_swap", "series_data_swap", "axis_scale_error"]):
        return "confusion_heatmap"
    chart = CHARTS[index % len(CHARTS)]
    if chart == "confusion_heatmap" and experiment_type != "multiclass":
        chart = "multi_panel"
    if chart in {"line", "scatter", "error_bar"} and experiment_type not in {"repeated_runs", "ablation"}:
        chart = "bar" if chart != "error_bar" else "grouped_bar"
    return chart


def category_plan() -> dict[str, list[list[str]]]:
    counts = {
        "dev": (8, 15, 7),
        "test_seen": (11, 23, 11),
        "test_unseen": (11, 22, 12),
    }
    singles = [e for e in ERROR_TYPES for _ in range(5)]
    combo_patterns = [
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
    ] * 3
    result, si, ci = {}, 0, 0
    for split, (clean_n, single_n, combo_n) in counts.items():
        plan = [[] for _ in range(clean_n)]
        plan += [[singles[i]] for i in range(si, si + single_n)]
        si += single_n
        plan += combo_patterns[ci:ci + combo_n]
        ci += combo_n
        rng = np.random.default_rng(SPLIT_SEED_BASE[split] + 777)
        rng.shuffle(plan)
        result[split] = plan
    return result


def main() -> None:
    data_root = ROOT / "data_v2"
    output_root = ROOT / "outputs" / "v2"
    for path in (data_root / "agent_view", data_root / "evaluator_view", output_root):
        path.mkdir(parents=True, exist_ok=True)
    plans = category_plan()
    split_manifest = {"version": VERSION, "splits": {}, "isolation": {}}
    all_records = []
    global_index = 0
    for split, plan in plans.items():
        refs = []
        for local_index, errors in enumerate(plan):
            seed = SPLIT_SEED_BASE[split] + local_index * 997 + 13
            rng = np.random.default_rng(seed)
            experiment_type = compatible_experiment(errors, global_index)
            model_count = 4 if experiment_type == "ablation" else int(rng.integers(2, 6))
            models = choose_models(rng, model_count, experiment_type)
            raw, summary, metric = make_raw(seed, experiment_type, models)
            chart = compatible_chart(errors, global_index, experiment_type)
            style = style_for(split, seed)
            template_index = local_index % len(TEMPLATES[split])
            table = clean_table(summary, experiment_type, metric)
            claim = make_claim(summary, metric, split, template_index, experiment_type)
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
                "case_id": cid, "experiment_type": experiment_type, "primary_metric": metric,
                "metric_directions": {metric: "minimize" if metric in {"MAE", "RMSE"} else "maximize"},
                "model_count": len(models), "chart_type": chart, "render_config": style,
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
                "summary": summary, "primary_metric": metric, "models": models,
                "plot_spec": plot_spec, "observed_plot_spec": observed_spec,
            })
            write_json(evaluator / "truth.json", {"case_id": cid, "split": split, "errors": truth})
            record = {
                "case_id": cid, "split": split, "experiment_type": experiment_type, "chart_type": chart,
                "seed": seed, "errors": truth,
                "paths": {"agent": f"agent_view/{split}/{cid}", "evaluator": f"evaluator_view/{split}/{cid}"},
                "versions": {"generator": VERSION, "injector": VERSION},
                "template_fingerprint": hashlib.sha256(TEMPLATES[split][template_index].encode()).hexdigest()[:12],
                "style_fingerprint": hashlib.sha256(json.dumps(style, sort_keys=True).encode()).hexdigest()[:12],
            }
            write_json(evaluator / "task.json", record)
            refs.append(cid)
            all_records.append(record)
            global_index += 1
        split_manifest["splits"][split] = refs
    split_manifest["isolation"] = {
        "seed_ranges": {split: [min(r["seed"] for r in all_records if r["split"] == split),
                                max(r["seed"] for r in all_records if r["split"] == split)] for split in SPLIT_SIZES},
        "template_fingerprints": {split: sorted({r["template_fingerprint"] for r in all_records if r["split"] == split}) for split in SPLIT_SIZES},
        "style_fingerprints": {split: sorted({r["style_fingerprint"] for r in all_records if r["split"] == split}) for split in SPLIT_SIZES},
    }
    write_json(data_root / "split_manifest.json", split_manifest)
    summary = {
        "task_count": len(all_records),
        "by_split": dict(Counter(r["split"] for r in all_records)),
        "by_error_count": dict(Counter(len(r["errors"]) for r in all_records)),
        "by_experiment_type": dict(Counter(r["experiment_type"] for r in all_records)),
        "by_chart_type": dict(Counter(r["chart_type"] for r in all_records)),
        "by_error_type": dict(Counter(e["error_type"] for r in all_records for e in r["errors"])),
    }
    write_json(output_root / "benchmark_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
