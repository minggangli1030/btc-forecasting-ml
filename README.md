# Bitcoin Price Forecasting with Machine Learning

This project uses machine learning and statistical models to forecast Bitcoin prices based on historical price data and external market indicators. Built as a final project for UC Berkeley's **Data 198: Introduction to Real-World Data Science** (Spring 2025 DeCal).

**Course Website:** [https://dssdecal.org/sp25/](https://dssdecal.org/sp25/)

## Models Implemented

- **ARIMA** (AutoRegressive Integrated Moving Average)
- **XGBoost** (Extreme Gradient Boosting Regression)

## Project Structure

```
btc-forecasting-ml/
├── btc-arima/          ← ARIMA models and data preprocessing
├── btc-xgboost/        ← XGBoost pipeline and lagged feature engineering
├── README.md
└── .gitignore
```

## Data Requirements

**Note:** Due to file size limitations, datasets are not included in this repository. Please download them manually and place them in the appropriate folders:

### Dataset 1: Bitcoin Historical Prices
- **Source:** [Kaggle — Cryptocurrency Historical Prices](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory)
- **Expected path:** `btc-arima/btc-historical-data.csv`

### Dataset 2: Bitcoin Market Sentiment Data
- **Source:** [Kaggle — Bitcoin Pulse: Market Trends & Fear Data](https://www.kaggle.com/datasets/wlwwwlw/bitcoin-pulse-market-trends-and-fear-dataset)
- **Expected path:** `btc-xgboost/btc-pulse-data.csv`

## Team & Credits

**Team:** Martin Li, Louis Liu, He Song, Natalie Wong  
**Mentor:** Alex Zhai

**Project Lead:** Minggang (Martin) Li  
UC Berkeley, Class of 2028  
[GitHub](https://github.com/minggangli1030)

## Course Information

Originally developed for the Spring 2025 **Data 198: Introduction to Real-World Data Science (DeCal)**  
**Course Website:** [https://dssdecal.org/sp25/](https://dssdecal.org/sp25/)

## Future Development

This repository will continue to evolve over the summer as an extended personal project.

## Repository

**GitHub:** [https://github.com/minggangli1030/btc-forecasting-ml](https://github.com/minggangli1030/btc-forecasting-ml)
