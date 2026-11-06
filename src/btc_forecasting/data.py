"""Explicit CSV schema and daily target construction."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_daily_prices(path: Path, timestamp: str, target: str) -> pd.Series:
    frame = pd.read_csv(path)
    missing = {timestamp, target} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    raw_time = frame[timestamp]
    dates = pd.to_datetime(raw_time, unit="s" if pd.api.types.is_numeric_dtype(raw_time) else None,
                           utc=True, errors="coerce")
    values = pd.to_numeric(frame[target], errors="coerce")
    if dates.isna().any():
        raise ValueError("Input contains invalid timestamps")
    if (values.dropna() <= 0).any() or np.isinf(values.dropna()).any():
        raise ValueError("Observed prices must be finite and positive")
    series = pd.Series(values.to_numpy(), index=dates, name="price").sort_index()
    series = series[~series.index.duplicated(keep="last")]
    daily = series.resample("1D").last().dropna()
    if len(daily) < 30:
        raise ValueError("At least 30 observed daily prices are required")
    # Row lags represent days only when the observation calendar is complete.
    if not daily.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Daily prices contain calendar gaps; resolve them explicitly before evaluation")
    return daily
