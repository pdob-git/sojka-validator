from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate classification metrics for a single category.

    Args:
        y_true: Ground truth labels (binary).
        y_score: Prediction scores (0-1).
        threshold: Classification threshold.

    Returns:
        Dictionary with precision, recall, f1, roc_auc, pr_auc, accuracy.
    """
    y_pred = (y_score >= threshold).astype(int)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def calculate_fpr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate False Positive Rate.

    Args:
        y_true: Ground truth labels (binary).
        y_pred: Predicted labels (binary).

    Returns:
        False Positive Rate.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1, 1):
        if cm[0, 0] == 0:
            return 0.0
        return 0.0 if y_pred.sum() == 0 else 1.0
    tn, fp, fn, tp = cm.ravel()
    if fp + tn == 0:
        return 0.0
    return float(fp / (fp + tn))


def calculate_all_metrics(
    df: Any,
    categories: list[str],
    thresholds: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Calculate metrics for all categories.

    Args:
        df: DataFrame with ground truth and predictions.
        categories: List of category columns.
        thresholds: Dictionary mapping category to threshold.

    Returns:
        Dictionary mapping category to metrics dictionary.
    """
    results = {}

    for category in categories:
        y_true = df[category].values
        y_score = df[f"{category}_pred"].values
        threshold = thresholds.get(category, 0.5)

        metrics = calculate_metrics(y_true, y_score, threshold)
        y_pred = (y_score >= threshold).astype(int)
        metrics["fpr"] = calculate_fpr(y_true, y_pred)

        results[category] = metrics

    return results


def precision_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.9,
) -> float | None:
    """Calculate precision at a given recall level.

    Args:
        y_true: Ground truth labels.
        y_score: Prediction scores.
        target_recall: Target recall level.

    Returns:
        Precision at target recall, or None if not achievable.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    idx = np.where(recall >= target_recall)[0]
    if len(idx) == 0:
        return None

    best_idx = idx[np.argmax(precision[idx])]
    return precision[best_idx]
