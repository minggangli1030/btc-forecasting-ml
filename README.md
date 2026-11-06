# Bitcoin Forecasting · Chronological ML Experiments

**Python · time-series evaluation · ARIMA · XGBoost · reproducible data workflows**

A UC Berkeley Data 198 course project comparing statistical and ML approaches to price forecasting. The original work did **not establish a reliable forecasting advantage**. This cleaned repository focuses on the engineering and evaluation lessons: explicit targets, chronological splits, past-only features, and comparison against a last-observed-price baseline.

## What is implemented

- **Validated data intake:** explicit timestamp/target columns, sorted UTC timestamps, last observed daily price, positive-price checks, and calendar-gap detection.
- **Temporal feature contracts:** lagged prices and trailing statistics built only from observations before the prediction date.
- **Aligned comparison:** persistence, ARIMA, and XGBoost evaluated on the same one-step-ahead target dates.
- **Repeatable experiments:** a command-line entry point writes dated predictions and RMSE/MAE reports; each run starts with fresh model state.
- **Offline checks:** synthetic tests verify that future prices cannot affect earlier features, splits remain chronological, and repeated runs do not mutate training data.

## Quick start — fictional data

Use Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
btc-forecast --demo
python -m pytest -q
```

The demo uses fictional seeded prices and the persistence baseline. Its output checks software behavior; it is not evidence of Bitcoin forecasting performance.

## Compare models on an authorized local dataset

```bash
python -m pip install -e '.[models,dev]'
btc-forecast --csv data/btc-historical-data.csv   --timestamp Timestamp --target Close --test-start 2024-01-01   --models naive arima xgboost --output outputs/comparison
```

`Timestamp` may be a UNIX timestamp in seconds or a parseable date string. `Close` is used consistently as the prediction target. Daily aggregation uses the last observed value, not the first intraday row. The loader rejects missing daily observations instead of silently treating multi-day gaps as single-day lags.

Original notebook sources: [Bitcoin historical prices](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) and [Bitcoin Pulse market data](https://www.kaggle.com/datasets/wlwwwlw/bitcoin-pulse-market-trends-and-fear-dataset). Downloads and outputs are excluded from Git. The cleaned benchmark intentionally uses price-derived features only; original external-market feature explorations remain in the historical notebook.

## Evaluation contract

Predict the price at date *t* using information observed before *t*. Training ends before the chosen cutoff. During testing, earlier observed test prices can enter later lag features or ARIMA state **after** their own forecast, matching rolling one-step evaluation. XGBoost is fitted once on training data; ARIMA parameters are fitted on training history and its state is updated without refitting. This is different from forecasting an entire unseen horizon without observing intervening prices.

No real-data benchmark was reproduced during cleanup, and no profitability or best-model claim is made. Before drawing conclusions, run the same holdout protocol, compare every model to persistence, and report error across multiple predeclared windows.

## Code map

| Path | Purpose |
| --- | --- |
| [src/btc_forecasting/data.py](src/btc_forecasting/data.py) | CSV and daily-price contract |
| [src/btc_forecasting/features.py](src/btc_forecasting/features.py) | Past-only features and chronological split |
| [src/btc_forecasting/evaluation.py](src/btc_forecasting/evaluation.py) | Baseline/model prediction and metrics |
| [src/btc_forecasting/cli.py](src/btc_forecasting/cli.py) | Reproducible experiment command |
| [tests/](tests/) | Synthetic temporal checks |
| [archive/notebooks/](archive/notebooks/) | Original team notebooks with outputs cleared |
| [docs/LEGACY-REVIEW.md](docs/LEGACY-REVIEW.md) | Original implementation issues and limitations |

## Team and provenance

Originally developed for UC Berkeley’s Spring 2025 **Data 198: Introduction to Real-World Data Science (DeCal)**. Original team credit: **Martin Li, Louis Liu, He Song, Natalie Wong**; mentor: **Alex Zhai**. Mentor-authored sections remain credited in the archived notebook. The cleanup adds a separate reproducible implementation and does not relabel the original team project as sole-authored work.

[Course website](https://dssdecal.org/sp25/) · [Validation](docs/VALIDATION.md)
