from __future__ import annotations

import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "matplotlib-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_plotting_data(metrics: dict, task_type: str) -> dict:
    primary = "Macro-F1" if task_type == "binary_classification" else "RMSE"
    return {
        "metric": primary,
        "labels": ["Model A", "Model B"],
        "values": [metrics["Model A"][primary], metrics["Model B"][primary]],
        "higher_is_better": task_type == "binary_classification",
    }


def render_bar_chart(plot_data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    bars = ax.bar(plot_data["labels"], plot_data["values"], color=["#4C78A8", "#F58518"])
    ax.set_ylabel(plot_data["metric"])
    ax.set_title(f"Model performance: {plot_data['metric']}")
    upper = max(plot_data["values"]) * 1.25
    ax.set_ylim(0, upper if upper > 0 else 1)
    for bar, value in zip(bars, plot_data["values"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": "ArtifactAudit"})
    plt.close(fig)
