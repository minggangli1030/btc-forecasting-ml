# 🧠 BTC Forecasting with Machine Learning

This project uses machine learning and statistical models to forecast Bitcoin prices based on historical price data and external market indicators. Built as a final project for UC Berkeley’s **Data 198: Introduction to Real-World Data Science (Spring 2025 DeCal)**.

---

## 📈 Models Used

- **ARIMA** (AutoRegressive Integrated Moving Average)
- **XGBoost** (Extreme Gradient Boosting Regression)

---

## 🗂️ Project Structure

```
btc-forecasting-ml/
├── btc-arima/           ← ARIMA models and data preprocessing
├── btc-xgboost/         ← XGBoost pipeline and lagged feature engineering
├── README.md
└── .gitignore
```

---

## 📂 Data Sources

⚠️ **Large CSV files are not included in this repository** due to GitHub’s file size limits.  
Please download them manually and place them in the appropriate folders:

### 🔹 Bitcoin Historical Data
- **Source:** [Kaggle — Cryptocurrency Historical Prices](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory)
- **Expected path:**  
  `btc-arima/btc-historical-data.csv`

### 🔹 Bitcoin Pulse Dataset
- **Source:** [Kaggle — Bitcoin Pulse: Market Trends & Fear Data](https://www.kaggle.com/datasets/wwwkaggler/bitcoin-pulse-market-trends-and-fear-dataset)
- **Expected path:**  
  `btc-xgboost/btc-pulse-data.csv`

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/minggangli1030/btc-forecasting-ml.git
cd btc-forecasting-ml

# Set up your Python environment
pip install -r requirements.txt

# (Optional) Run notebooks with Jupyter or VSCode
```

---

## 🧪 Author

Minggang (Martin) Li  
UC Berkeley, Class of 2028  
[GitHub](https://github.com/minggangli1030)

---

## 📜 License

MIT License — feel free to use, share, or adapt the code with attribution.
