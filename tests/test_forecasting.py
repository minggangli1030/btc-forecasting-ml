"""Temporal contracts on fictional prices, not BTC-performance claims."""
import numpy as np
import pandas as pd
import pytest
from btc_forecasting.data import load_daily_prices
from btc_forecasting.features import make_features, chronological_split
from btc_forecasting.evaluation import evaluate


@pytest.fixture
def prices():
    return pd.Series(np.linspace(100, 180, 80), index=pd.date_range("2024-01-01", periods=80, tz="UTC"))


def test_future_prices_cannot_change_earlier_features(prices):
    altered = prices.copy()
    altered.iloc[50:] *= 10
    pd.testing.assert_frame_equal(make_features(prices).iloc[:51], make_features(altered).iloc[:51])


def test_cutoff_and_training_rows_are_held_fixed(prices):
    train, test = chronological_split(prices, "2024-02-15")
    assert train.index.max() < test.index.min()
    altered = prices.copy()
    altered.loc[test.index] *= 2
    pd.testing.assert_frame_equal(train, chronological_split(altered, "2024-02-15")[0])


def test_naive_forecast_uses_previous_observation(prices):
    frame, metrics = evaluate(prices, "2024-02-15")
    np.testing.assert_allclose(frame.naive, prices.shift(1).loc[frame.index])
    assert metrics['naive']['rmse'] == pytest.approx(80 / 79)


def test_repeated_runs_do_not_mutate_training_history(prices):
    snapshot = prices.copy()
    first, _ = evaluate(prices, "2024-02-15")
    second, _ = evaluate(prices, "2024-02-15")
    pd.testing.assert_series_equal(prices, snapshot)
    pd.testing.assert_frame_equal(first, second)


def test_daily_loader_sorts_and_takes_last_observed_price(tmp_path):
    times = pd.date_range('2024-01-01', periods=60, freq='12h', tz='UTC')
    path = tmp_path / 'fictional.csv'
    pd.DataFrame({'Timestamp': times, 'Close': np.arange(60) + 1}).iloc[::-1].to_csv(path, index=False)
    prices = load_daily_prices(path, 'Timestamp', 'Close')
    assert len(prices) == 30 and prices.iloc[0] == 2 and prices.iloc[-1] == 60


def test_daily_loader_rejects_calendar_gaps(tmp_path):
    dates = pd.date_range('2024-01-01', periods=40, tz='UTC').delete(12)
    path = tmp_path / 'fictional.csv'
    pd.DataFrame({'Timestamp': dates, 'Close': np.arange(39) + 1}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='calendar gaps'):
        load_daily_prices(path, 'Timestamp', 'Close')


def test_rejects_empty_split_and_unknown_model(prices):
    with pytest.raises(ValueError):
        evaluate(prices, '2030-01-01')
    with pytest.raises(ValueError):
        evaluate(prices, '2024-02-15', ('unknown',))
