from __future__ import annotations

import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "matplotlib-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FONTS = ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"]
PALETTES = [
    ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"],
    ["#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1", "#955251"],
    ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"],
]


def style_for(split: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    unseen = split == "test_unseen"
    return {
        "palette": PALETTES[int(rng.integers(1, 3)) if unseen else int(rng.integers(0, 2))],
        "font": FONTS[2] if unseen else FONTS[int(rng.integers(0, 2))],
        "dpi": int(rng.choice([110, 140, 180] if unseen else [90, 100, 120])),
        "figsize": [float(v) for v in rng.choice([[8, 4], [5, 6], [9, 5]] if unseen else [[6.4, 4.8], [7, 4], [5, 5]])],
        "legend_loc": str(rng.choice(["upper left", "upper right", "lower center"] if unseen else ["best", "upper right", "lower left"])),
        "legend_outside": bool(rng.integers(0, 2)),
        "grid": bool(rng.integers(0, 2)),
        "horizontal": bool(rng.integers(0, 2)),
        "precision": int(rng.choice([2, 3, 4])),
        "marker": str(rng.choice(["D", "^", "s"] if unseen else ["o", "s", "x"])),
    }


def render_plot(spec: dict, path: Path) -> None:
    style = spec["style"]
    plt.rcParams.update({"font.family": style["font"]})
    fig, ax = plt.subplots(figsize=style["figsize"], dpi=style["dpi"])
    labels, values = spec["labels"], spec["values"]
    colors = style["palette"][:len(labels)]
    chart = spec["chart_type"]
    if chart in {"bar", "grouped_bar", "error_bar"}:
        positions = np.arange(len(labels))
        errors = spec.get("errors")
        if style["horizontal"]:
            ax.barh(positions, values, xerr=errors, color=colors)
            ax.set_yticks(positions, labels)
            ax.set_xlabel(spec["metric"])
        else:
            ax.bar(positions, values, yerr=errors, color=colors)
            ax.set_xticks(positions, labels, rotation=15)
            ax.set_ylabel(spec["metric"])
    elif chart == "line":
        x = np.arange(len(spec["series_x"]))
        for i, label in enumerate(labels):
            ax.plot(x, spec["series"][i], marker=style["marker"], color=colors[i], label=label)
        ax.set_xticks(x, spec["series_x"])
        ax.set_ylabel(spec["metric"])
    elif chart == "scatter":
        for i, label in enumerate(labels):
            ax.scatter(spec["scatter_x"][i], spec["scatter_y"][i], color=colors[i], label=label, s=35)
        ax.set_xlabel("Run index")
        ax.set_ylabel(spec["metric"])
    elif chart == "confusion_heatmap":
        matrix = np.asarray(spec["matrix"])
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    elif chart == "multi_panel":
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=style["figsize"], dpi=style["dpi"])
        axes[0].bar(labels, values, color=colors)
        axes[0].set_ylabel(spec["metric"])
        axes[1].plot(range(len(values)), values, marker=style["marker"], color=colors[0])
        axes[1].set_xticks(range(len(labels)), labels, rotation=15)
        ax = axes[0]
    ax.set_title(spec["title"])
    if style["grid"]:
        ax.grid(alpha=0.25)
    if chart in {"line", "scatter"}:
        if style["legend_outside"]:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(loc=style["legend_loc"])
    if spec.get("axis_limits"):
        if style["horizontal"] and chart in {"bar", "grouped_bar", "error_bar"}:
            ax.set_xlim(*spec["axis_limits"])
        else:
            ax.set_ylim(*spec["axis_limits"])
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": "ArtifactAuditBenchV2"})
    plt.close(fig)
