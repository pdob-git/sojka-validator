from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sojka_validator.data_loader import load_benchmark_data
from sojka_validator.inference import BielikGuardPredictor
from sojka_validator.metrics import calculate_all_metrics, precision_at_recall
from sojka_validator.threshold import find_thresholds_for_all_categories


def run_benchmark(
    data_path: str | Path,
    model_name: str = "speakleash/bielik-guard",
    categories: list[str] | None = None,
    device: int = -1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run full benchmark evaluation on Bielik Guard model.

    Args:
        data_path: Path to benchmark Excel file.
        model_name: HuggingFace model name.
        categories: List of category columns to evaluate.
        device: Device to use (-1 for CPU, 0+ for GPU).
        show_progress: Whether to show progress bars.

    Returns:
        DataFrame with metrics for each category.
    """
    if categories is None:
        categories = ["self-harm", "hate", "sex", "crime"]

    df = load_benchmark_data(data_path, categories=categories)

    predictor = BielikGuardPredictor(model_name=model_name, device=device)
    df = predictor.predict_dataframe(df, show_progress=show_progress)

    thresholds = find_thresholds_for_all_categories(df, categories)

    results = calculate_all_metrics(df, categories, thresholds)

    report = pd.DataFrame(results).T
    report["threshold"] = [thresholds[cat] for cat in report.index]

    cols = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "fpr",
        "accuracy",
    ]
    report = report[cols]

    return report


def run_benchmark_with_detailed_analysis(
    data_path: str | Path,
    model_name: str = "speakleash/bielik-guard",
    categories: list[str] | None = None,
    device: int = -1,
    target_recall: float = 0.9,
) -> dict[str, Any]:
    """Run benchmark with detailed analysis including precision@recall.

    Args:
        data_path: Path to benchmark Excel file.
        model_name: HuggingFace model name.
        categories: List of category columns to evaluate.
        device: Device to use (-1 for CPU, 0+ for GPU).
        target_recall: Target recall for precision@recall calculation.

    Returns:
        Dictionary with detailed results.
    """
    if categories is None:
        categories = ["self-harm", "hate", "sex", "crime"]

    df = load_benchmark_data(data_path, categories=categories)

    predictor = BielikGuardPredictor(model_name=model_name, device=device)
    df = predictor.predict_dataframe(df, show_progress=True)

    thresholds = find_thresholds_for_all_categories(df, categories)

    results = calculate_all_metrics(df, categories, thresholds)

    precision_at_recall_values = {}
    for category in categories:
        y_true = np.asarray(df[category].values, dtype=float)
        y_score = np.asarray(df[f"{category}_pred"].values, dtype=float)
        precision_at_recall_values[category] = precision_at_recall(
            y_true, y_score, target_recall
        )

    return {
        "metrics": pd.DataFrame(results).T,
        "thresholds": thresholds,
        "precision_at_recall": precision_at_recall_values,
        "data": df,
    }
