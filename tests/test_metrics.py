import numpy as np

from src.metrics import classification_metrics, regression_metrics


def test_classification_metrics_correct():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.8, 0.9])
    assert np.isclose(result["Accuracy"], 0.75)
    assert np.isclose(result["Precision"], 2 / 3)
    assert np.isclose(result["Recall"], 1.0)
    assert np.isclose(result["Macro-F1"], (2 / 3 + 0.8) / 2)
    assert np.isclose(result["ROC-AUC"], 1.0)


def test_regression_metrics_correct():
    result = regression_metrics([1, 2, 3], [1, 2, 5])
    assert np.isclose(result["MAE"], 2 / 3)
    assert np.isclose(result["RMSE"], np.sqrt(4 / 3))
    assert np.isclose(result["R2"], -1.0)
