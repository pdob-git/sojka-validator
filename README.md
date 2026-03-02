# Sojka Validator

Validator for Bielik Sójka benchmark system - evaluates safety/guard models for Polish language content moderation.

## What It Does

This tool evaluates the **Bielik Guard** model (a Polish safety classifier) on benchmark data to measure its performance at detecting:
- **self-harm** - Self-harm content
- **hate** - Hate speech
- **sex** - Sexual content
- **crime** - Criminal content

It calculates metrics like F1, ROC AUC, PR AUC, precision, recall, and optimizes classification thresholds.

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root with the following variables:

```bash
MODEL_NAME=speakleash/Bielik-Guard-0.1B-v1.1
BENCHMARK_DATA=data/full_benchmark.xlsx
```

- `MODEL_NAME`: HuggingFace model name (e.g., `speakleash/Bielik-Guard-0.1B-v1.1` or `speakleash/Bielik-Guard-0.5B-v1.1`)
- `BENCHMARK_DATA`: Path to the benchmark Excel file

## Quick Start

Run benchmark with test data:

```bash
python -m sojka_validator
```

Or programmatically:

```python
from sojka_validator import run_benchmark

# Run on test data
results = run_benchmark("data/full_benchmark.xlsx")
print(results)
```

## Usage Examples

### 1. Basic Benchmark

```python
from sojka_validator import run_benchmark

results = run_benchmark(
    data_path="data/full_benchmark.xlsx",
    model_name="speakleash/bielik-guard",
    device=-1,  # -1 for CPU, 0+ for GPU
)
print(results)
```

### 2. Detailed Analysis with Precision@Recall

```python
from sojka_validator import run_benchmark_with_detailed_analysis

results = run_benchmark_with_detailed_analysis(
    data_path="data/full_benchmark.xlsx",
    target_recall=0.9,  # Calculate precision at 90% recall
)

# Access different parts
print("Metrics:", results["metrics"])
print("Optimal thresholds:", results["thresholds"])
print("Precision@90% recall:", results["precision_at_recall"])
```

### 3. Custom Categories

```python
from sojka_validator import run_benchmark

results = run_benchmark(
    data_path="data/full_benchmark.xlsx",
    categories=["self-harm", "hate", "sex", "crime"],
)
```

### 4. Using the Predictor Directly

```python
from sojka_validator import BielikGuardPredictor

# Initialize predictor
predictor = BielikGuardPredictor(
    model_name="speakleash/bielik-guard",
    device=-1,
)

# Single prediction
scores = predictor.predict("To jest normalna wiadomość")
print(scores)

# Batch predictions
texts = ["Text 1", "Text 2", "Text 3"]
predictions = predictor.predict_batch(texts)
```

## Output

The benchmark returns a DataFrame with these metrics per category:

| Metric | Description |
|--------|-------------|
| threshold | Optimized classification threshold (maximizes F1) |
| precision | Precision score |
| recall | Recall score |
| f1 | F1 score |
| roc_auc | ROC AUC score |
| pr_auc | Precision-Recall AUC |
| fpr | False Positive Rate |
| accuracy | Accuracy |

## Benchmark Data Format

Your Excel file (`data/full_benchmark.xlsx`) should have:
- A column named `fraza` containing the text to analyze
- Binary columns (0/1) for each category: `self-harm`, `hate`, `sex`, `crime`

## Model Variants

Bielik Guard comes in different sizes:

- **0.1B** (124M parameters) - Faster, more efficient
  - `speakleash/Bielik-Guard-0.1B-v1.1`
- **0.5B** (443M parameters) - Higher accuracy
  - `speakleash/Bielik-Guard-0.5B-v1.1`

Default: `speakleash/bielik-guard` (uses the best available)
