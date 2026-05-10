from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpc_carbon_scheduler.carbon_model import load_and_prepare_jobs, train_energy_model
from hpc_carbon_scheduler.plotting import save_energy_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train energy/carbon prediction model.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "job_table.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "carbon_prediction")
    parser.add_argument("--max-rows", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.input}\n"
            "Place the PM100 parquet file at data/job_table.parquet, "
            "or pass a custom path with --input path/to/job_table.parquet."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_and_prepare_jobs(args.input, max_rows=args.max_rows, seed=args.seed)
    artifacts = train_energy_model(jobs, random_state=args.seed)

    metrics = pd.DataFrame([artifacts.metrics])
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    artifacts.predictions.to_csv(args.output_dir / "job_predictions.csv", index=False)
    artifacts.feature_importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    save_energy_plots(artifacts.predictions, artifacts.feature_importance, args.output_dir)

    print("Carbon/energy prediction completed")
    print(metrics.to_string(index=False))
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
