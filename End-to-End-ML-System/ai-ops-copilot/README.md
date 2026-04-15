# AI Ops Copilot – Energy Demand Forecasting

## 📌 Overview
This project builds an end-to-end machine learning system for forecasting energy consumption using time-series data.

## 🚀 Features
- Time-series forecasting using XGBoost
- Advanced feature engineering (lags, rolling stats, cyclical encoding)
- Holiday-based demand modeling
- Proper time-based validation (last-year split)
- EDA and data quality analysis

## 📊 Dataset
PJM Energy Consumption Dataset (Kaggle)

## 🧠 Approach
- Data cleaning and preprocessing
- Feature engineering:
  - Lag features (1, 24, 168)
  - Rolling statistics
  - Time-based features
- Model:
  - XGBoost Regressor

## 📈 Results
- MAE: <your result>

## 📂 Project Structure
(mention structure)

## ▶️ Run
```bash
pip install -r requirements.txt
python run_training.py