import os
from pathlib import Path

from dotenv import load_dotenv

from sojka_validator import run_benchmark


def main() -> None:
    """Run benchmark using configuration from .env file."""
    load_dotenv()

    model_name = os.getenv("MODEL_NAME")
    benchmark_data = os.getenv("BENCHMARK_DATA")

    if not model_name:
        raise ValueError("MODEL_NAME not found in .env file")
    if not benchmark_data:
        raise ValueError("BENCHMARK_DATA not found in .env file")

    benchmark_path = Path(benchmark_data)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_data}")

    print("Running benchmark...")
    print(f"  Model: {model_name}")
    print(f"  Data: {benchmark_data}")
    print()

    results = run_benchmark(
        data_path=benchmark_path,
        model_name=model_name,
    )

    print("\nResults:")
    print(results.to_string())


if __name__ == "__main__":
    main()
