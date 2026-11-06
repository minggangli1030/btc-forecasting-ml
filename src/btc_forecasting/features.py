"""Features at date t use only prices observed before t."""
import pandas as pd


def make_features(prices: pd.Series) -> pd.DataFrame:
    past = prices.shift(1)
    return pd.DataFrame({
        "lag_1": past,
        "lag_2": prices.shift(2),
        "lag_7": prices.shift(7),
        "mean_7": past.rolling(7).mean(),
        "std_7": past.rolling(7).std(),
    }, index=prices.index)


def chronological_split(prices: pd.Series, test_start: str):
    frame = make_features(prices).assign(target=prices).dropna()
    cutoff = pd.Timestamp(test_start)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    train = frame.loc[frame.index < cutoff]
    test = frame.loc[frame.index >= cutoff]
    if len(train) < 20 or test.empty:
        raise ValueError("Choose a cutoff with at least 20 usable training rows and one test row")
    return train, test
