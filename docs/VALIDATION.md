# Validation

Prepared November 6, 2026.

- 7 synthetic temporal-contract tests passed.
- The synthetic CLI ran persistence, ARIMA, and XGBoost on the same 28 fictional test dates and wrote aligned predictions and RMSE/MAE reports.
- ARIMA emitted starting-parameter warnings on the fictional series; the run completed. Synthetic scores are software smoke tests and are not published as real-data model performance.
- Python syntax, local Markdown links, and notebook-output clearing were checked.

No original Bitcoin dataset was downloaded, historical result reproduced, profitable strategy established, or real-market forecasting improvement claimed. Original team/mentor credits remain intact. The historical notebooks are retained with known evaluation issues documented rather than represented as the cleaned benchmark.

The first clean GitHub runner exposed that the XGBoost sklearn adapter requires scikit-learn explicitly. It is now declared in the models dependency group; the model smoke-test remains enabled.
