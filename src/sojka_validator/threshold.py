from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_thresholds: int = 100,
) -> tuple[float, float]:
    """Find the best threshold for F1 score.

    Args:
        y_true: Ground truth labels (binary).
        y_score: Prediction scores (0-1).
        n_thresholds: Number of thresholds to try.

    Returns:
        Tuple of (best_threshold, best_f1).
    """
    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def find_thresholds_for_all_categories(
    df: Any,
    categories: list[str],
) -> dict[str, float]:
    """Find optimal threshold for each category.

    Args:
        df: DataFrame with ground truth and predictions.
        categories: List of category columns.

    Returns:
        Dictionary mapping category to optimal threshold.
    """
    thresholds = {}

    for category in categories:
        y_true = df[category].values
        y_score = df[f"{category}_pred"].values

        best_threshold, best_f1 = find_best_threshold(y_true, y_score)
        thresholds[category] = best_threshold

    return thresholds


def find_threshold_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.9,
) -> float | None:
    """Find threshold that achieves at least target recall.

    Args:
        y_true: Ground truth labels.
        y_score: Prediction scores.
        target_recall: Target recall level.

    Returns:
        Threshold achieving target recall, or None if not achievable.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    idx = np.where(recall >= target_recall)[0]
    if len(idx) == 0:
        return None

    best_idx = idx[np.argmax(precision[idx])]
    return thresholds[best_idx]
