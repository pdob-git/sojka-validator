import pandas as pd
from transformers import Pipeline, pipeline


class BielikGuardPredictor:
    """Predictor for Bielik Guard model from HuggingFace."""

    def __init__(
        self,
        model_name: str = "speakleash/bielik-guard",
        device: int = -1,
    ) -> None:
        """Initialize the Bielik Guard predictor.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to use (-1 for CPU, 0+ for GPU).
        """
        self.model_name = model_name
        self.device = device
        self._pipeline: Pipeline | None = None

    @property
    def pipeline(self) -> Pipeline:
        """Lazy loading of the model pipeline."""
        if self._pipeline is None:
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=self.device,
            )
        return self._pipeline

    def predict(self, text: str) -> dict[str, float]:
        """Predict safety scores for a single text.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary mapping category to score (0-1).
        """
        results = self.pipeline(text)[0]
        return {r["label"]: r["score"] for r in results}

    def predict_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> dict[str, list[float]]:
        """Predict safety scores for multiple texts.

        Args:
            texts: List of input texts to analyze.
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary mapping category to list of scores.
        """
        predictions: dict[str, list[float]] = {}

        try:
            from tqdm import tqdm

            iterator: list[str] | tqdm = (
                tqdm(texts, desc="Running inference") if show_progress else texts
            )
        except ImportError:
            iterator = texts

        for text in iterator:
            result = self.predict(text)
            if not predictions:
                predictions = {k: [] for k in result.keys()}
            for k, v in result.items():
                predictions[k].append(v)

        return predictions

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Predict safety scores for all texts in a DataFrame.

        Args:
            df: DataFrame containing texts.
            text_column: Name of the column containing texts.
            show_progress: Whether to show progress bar.

        Returns:
            DataFrame with added prediction columns.
        """
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, show_progress=show_progress)

        result = df.copy()
        for category, scores in predictions.items():
            result[f"{category}_pred"] = scores

        return result
