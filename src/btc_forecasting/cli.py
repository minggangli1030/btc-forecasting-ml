"""Run synthetic checks or a local, chronological CSV benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_daily_prices
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--timestamp", default="Timestamp")
    parser.add_argument("--target", default="Close")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--models", nargs="+", choices=["naive", "arima", "xgboost"], default=["naive"])
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--demo", action="store_true", help="Use fictional seeded prices; no real BTC data")
    args = parser.parse_args()
    if args.demo == bool(args.csv):
        parser.error("Choose exactly one of --csv and --demo")
    if args.demo:
        rng = np.random.default_rng(42)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))),
                           index=pd.date_range("2023-10-01", periods=120, tz="UTC"), name="price")
    else:
        prices = load_daily_prices(args.csv, args.timestamp, args.target)
    predictions, metrics = evaluate(prices, args.test_start, tuple(args.models), args.max_test_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index_label="date")
    report = {"synthetic": args.demo, "target_column": args.target if not args.demo else "fictional_price",
              "task": "one_step_ahead_with_observed_history", "test_start": args.test_start,
              "feature_policy": "past_price_lags_only", "metrics": metrics}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
