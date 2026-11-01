from __future__ import annotations

from src.v2.core import METRIC_DIRECTIONS, rank_models, significance

PHRASES = [
    "achieves", "obtains", "outperforms", "yields", "improves upon",
    "reduces", "is consistently better than", "shows no statistically reliable gain",
    "contributes the largest improvement", "remains competitive with",
]

TEMPLATES = {
    "dev": [
        "{best} achieves the {direction} {metric} ({bestv:.3f}).",
        "{best} obtains the best {metric}: {bestv:.3f}.",
        "{best} outperforms {other} on {metric} by {abs:.3f}.",
        "{best} yields a {rel:.2f}% relative improvement in {metric}.",
        "{best} improves upon {other} by {pp:.2f} percentage points on {metric}.",
        "{best} is consistently better than {other} for {metric}.",
        "{best} remains competitive with {other} ({bestv:.3f} vs {otherv:.3f}).",
        "The {metric} leader is {best}, with an absolute gain of {abs:.3f}.",
        "{best} records {bestv:.3f} {metric}; {other} records {otherv:.3f}.",
        "On {metric}, {best} ranks first and {other} ranks second.",
    ],
    "test_seen": [
        "Across methods, {best} achieves the {direction} {metric} at {bestv:.3f}.",
        "The best observed {metric} is obtained by {best} ({bestv:.3f}).",
        "For {metric}, {best} outperforms {other}; the absolute gap is {abs:.3f}.",
        "The result from {best} yields a relative {metric} change of {rel:.2f}%.",
        "Compared with {other}, {best} improves {metric} by {pp:.2f} points.",
        "Evaluation on {metric} finds {best} consistently better than {other}.",
        "Although close to {other}, {best} remains competitive and leads {bestv:.3f} to {otherv:.3f}.",
        "An absolute margin of {abs:.3f} makes {best} the {metric} leader.",
        "Reported {metric}: {best}={bestv:.3f}, {other}={otherv:.3f}.",
        "Ordering systems by {metric} puts {best} ahead of {other}.",
    ],
    "test_unseen": [
        "The strongest {metric} is delivered by {best} at {bestv:.3f}.",
        "With {bestv:.3f} on {metric}, {best} leads the comparison.",
        "A margin of {abs:.3f} separates {best} from {other} on {metric}.",
        "Relative to {other}, {best} changes {metric} by {rel:.2f}%.",
        "The {metric} gap is {pp:.2f} points in favor of {best}.",
        "{other} trails {best} on the reported {metric}.",
        "{best} and {other} remain close ({bestv:.3f} versus {otherv:.3f}), with {best} ahead.",
        "First place on {metric} belongs to {best}; the margin is {abs:.3f}.",
        "Measured {metric} values are {bestv:.3f} for {best} and {otherv:.3f} for {other}.",
        "Ranking by {metric} places {best} before {other}.",
    ],
}


def claim_context(summary: dict, metric: str) -> dict:
    best, other = rank_models(summary, metric)[:2]
    bestv, otherv = summary[best][metric], summary[other][metric]
    absolute = abs(bestv - otherv)
    return {
        "best": best, "other": other, "metric": metric, "bestv": bestv, "otherv": otherv,
        "abs": absolute, "pp": absolute * 100, "rel": absolute / abs(otherv) * 100 if otherv else 0.0,
        "direction": "highest" if METRIC_DIRECTIONS.get(metric) == "maximize" else "lowest",
    }


def make_claim(summary: dict, metric: str, split: str, template_index: int, experiment_type: str) -> str:
    context = claim_context(summary, metric)
    base = TEMPLATES[split][template_index % len(TEMPLATES[split])].format(**context)
    if experiment_type == "repeated_runs":
        sig = significance(summary, context["best"], context["other"])
        suffix = (
            " The 95% confidence intervals do not overlap, indicating a statistically reliable difference."
            if sig else " The confidence intervals overlap, showing no statistically reliable gain."
        )
        base += suffix
    if experiment_type == "ablation":
        base += f" The full model contributes the largest improvement over {context['other']}."
    return base + "\n"
