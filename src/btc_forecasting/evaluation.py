"""Aligned one-step-ahead evaluation with a persistence baseline."""
import numpy as np
import pandas as pd
from .features import chronological_split


def arima_predictions(history: np.ndarray, observed_test: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    # A fresh model per run; held-out observations enter state only after prediction.
    fitted = ARIMA(np.log(history), order=(2, 1, 1)).fit()
    predictions = []
    for actual in observed_test:
        predictions.append(float(np.exp(np.asarray(fitted.forecast(1))[0])))
        fitted = fitted.append([float(np.log(actual))], refit=False)
    return np.asarray(predictions)


def evaluate(prices: pd.Series, test_start: str, models=("naive",), max_test_rows=None):
    if not models or len(set(models)) != len(models) or set(models) - {"naive", "arima", "xgboost"}:
        raise ValueError("Models must be a nonempty, unique selection of naive, arima, xgboost")
    if max_test_rows is not None and max_test_rows < 1:
        raise ValueError("max_test_rows must be positive")
    train, test = chronological_split(prices, test_start)
    if max_test_rows is not None:
        test = test.iloc[:max_test_rows]
    result = pd.DataFrame({"actual": test.target}, index=test.index)
    if "naive" in models:
        result["naive"] = test.lag_1
    if "arima" in models:
        history = prices.loc[prices.index < test.index[0]].to_numpy()
        result["arima"] = arima_predictions(history, test.target.to_numpy())
    if "xgboost" in models:
        from xgboost import XGBRegressor
        columns = [c for c in train.columns if c != "target"]
        model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                             random_state=42, n_jobs=1, objective="reg:squarederror")
        model.fit(train[columns], train.target)
        result["xgboost"] = model.predict(test[columns])
    metrics = {}
    for name in models:
        residual = result[name].to_numpy() - result.actual.to_numpy()
        if not np.isfinite(residual).all():
            raise ValueError(f"{name} produced nonfinite predictions")
        metrics[name] = {"rmse": float(np.sqrt(np.mean(residual ** 2))),
                         "mae": float(np.mean(np.abs(residual))), "n_test": len(test)}
    return result, metrics
