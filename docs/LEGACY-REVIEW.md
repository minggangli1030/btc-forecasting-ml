# Historical notebooks and lessons

The original notebooks are preserved under `archive/notebooks/` for provenance. Their code is historical and should not be used as the current evaluation entry point.

## Issues identified

1. The ARIMA notebook appends held-out observations to a mutable training list and then reuses that list for later experiments. Those later model fits can include previously accessed test observations.
2. Its multi-day section trains on changes in `Close` but reconstructs and scores against `High`; that is not a consistent target definition.
3. Daily input selection keeps the first row for each date rather than explicitly defining a daily closing observation.
4. The README's original ARIMA dataset link did not match the notebook's stated source.
5. No common persistence benchmark, repeatable environment, or automated temporal contracts were supplied.

The cleaned implementation uses one explicit daily target, a fresh model per run, strictly past-only features, and shared evaluation dates. It does not silently replace the original reported results with new claims. The original work did not establish meaningful forecasting gains, and the cleanup does not manufacture them.

## Remaining limitations

The cleaned runner uses fixed model settings and one chosen cutoff. It is not nested model selection, a full walk-forward hyperparameter study, or a trading backtest. Price persistence is a necessary baseline. External indicators require timestamp-availability checks before they can be added without look-ahead bias. Error metrics on prices do not establish tradable returns after costs.
