from __future__ import annotations

import numpy as np

GENERATOR_VERSION = "1.0.0"


def generate_binary_data(random_seed: int, n_samples: int = 240) -> list[dict]:
    rng = np.random.default_rng(random_seed)
    latent = rng.normal(0.0, 1.0, n_samples)
    y_true = (latent + rng.normal(0, 0.55, n_samples) > 0).astype(int)
    logits_a = 1.35 * latent + rng.normal(0, 0.85, n_samples)
    logits_b = 1.75 * latent + rng.normal(0, 0.65, n_samples)
    prob_a = 1.0 / (1.0 + np.exp(-logits_a))
    prob_b = 1.0 / (1.0 + np.exp(-logits_b))
    return [
        {
            "sample_id": i,
            "y_true": int(y_true[i]),
            "model_a_probability": float(prob_a[i]),
            "model_b_probability": float(prob_b[i]),
            "model_a_prediction": int(prob_a[i] >= 0.5),
            "model_b_prediction": int(prob_b[i] >= 0.5),
        }
        for i in range(n_samples)
    ]


def generate_regression_data(random_seed: int, n_samples: int = 240) -> list[dict]:
    rng = np.random.default_rng(random_seed)
    x = rng.uniform(-3.0, 3.0, n_samples)
    y_true = 1.8 * x - 0.35 * x**2 + rng.normal(0, 0.45, n_samples)
    pred_a = y_true + rng.normal(0.10, 0.95, n_samples)
    pred_b = y_true + rng.normal(-0.03, 0.60, n_samples)
    return [
        {
            "sample_id": i,
            "y_true": float(y_true[i]),
            "model_a_prediction": float(pred_a[i]),
            "model_b_prediction": float(pred_b[i]),
        }
        for i in range(n_samples)
    ]
