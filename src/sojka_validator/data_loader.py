from pathlib import Path

import pandas as pd


def load_benchmark_data(
    file_path: str | Path,
    text_column: str = "fraza",
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Load benchmark data from Excel file.

    Args:
        file_path: Path to the Excel file containing benchmark data.
        text_column: Name of the column containing text to analyze.
        categories: List of category columns to use. If None, uses
            ['self-harm', 'hate', 'sex', 'crime'].

    Returns:
        DataFrame with text and category columns.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    if categories is None:
        categories = ["self-harm", "hate", "sex", "crime"]

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    df = pd.read_excel(path)

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    missing_categories = [c for c in categories if c not in df.columns]
    if missing_categories:
        raise ValueError(
            f"Category columns not found: {missing_categories}. "
            f"Available columns: {df.columns.tolist()}"
        )

    for cat in categories:
        df[cat] = df[cat].fillna(0).astype(int)

    result = df[[text_column] + categories].copy()
    result.columns = ["text"] + categories

    return result


def add_predictions(
    df: pd.DataFrame,
    predictions: dict[str, list[float]],
) -> pd.DataFrame:
    """Add prediction scores to the dataframe.

    Args:
        df: DataFrame with ground truth labels.
        predictions: Dictionary mapping category to list of prediction scores.

    Returns:
        DataFrame with added prediction columns.
    """
    result = df.copy()
    for category, scores in predictions.items():
        result[f"{category}_pred"] = scores
    return result
