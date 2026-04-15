# ⚡ Hourly Energy Consumption Forecasting (PJM)

## 📌 Overview

This project focuses on forecasting hourly energy consumption using historical data from the PJM Interconnection grid.

The dataset spans over a decade (2002–2018) and exhibits strong temporal patterns such as:

* Daily consumption cycles
* Weekly variations (weekday vs weekend)
* Seasonal trends (summer and winter peaks)

The goal is to build a robust machine learning model that captures these patterns effectively.

---

## 🎯 Problem Statement

The objective of this project is to develop a time-series forecasting model to predict hourly energy consumption using historical data.

Accurate energy demand forecasting is essential for:

* Optimizing power generation
* Ensuring grid stability
* Reducing operational costs
* Supporting data-driven decision-making

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost

---

## 📊 Project Workflow

### 1. Data Understanding & Preprocessing

* Converted datetime column to proper format
* Sorted and indexed time-series data
* Removed duplicates and validated data quality

### 2. Exploratory Data Analysis (EDA)

* Visualized long-term trends
* Identified hourly, weekly, and seasonal patterns
* Analyzed distribution of energy consumption

### 3. Feature Engineering

* Time-based features:

  * Hour, Day of Week, Month, Year
* Lag features:

  * lag_1 (previous hour)
  * lag_24 (previous day)
  * lag_168 (previous week)
* Rolling statistics:

  * Rolling mean and standard deviation

These features help capture temporal dependencies in the data.

---

## 🤖 Modeling Approach

We compare multiple models:

* Linear Regression → Baseline model
* Random Forest → Captures non-linear patterns
* XGBoost → Final model (best performance)

XGBoost is chosen due to:

* Ability to model complex non-linear relationships
* High efficiency and scalability
* Strong performance on structured/tabular data

---

## 📈 Model Performance

| Model             | RMSE       |
| ----------------- | ---------- |
| Linear Regression | 1274.96    |
| Random Forest     | 610.88     |
| XGBoost           | **433.92** |

---

## 🧠 Key Insights

* Energy consumption is highly influenced by:

  * Time of day
  * Day of week
  * Seasonal variations
* Strong temporal dependencies exist in the data
* Feature engineering significantly improves model performance
* XGBoost outperforms traditional models for time-series forecasting

---

## 📊 Sample Output

The model successfully captures overall trends and seasonal patterns in energy consumption.

*(You can add your prediction plot here for better visualization)*

---

## ⚙️ Production Perspective

This solution can be extended into a real-world system using:

* Azure ML or cloud-based pipelines
* Automated retraining workflows
* Monitoring for data drift and performance

---

## 🔮 Future Improvements

* Hyperparameter tuning for XGBoost
* Deep learning models (LSTM, Transformers)
* Real-time forecasting pipelines

---

## 📂 Project Structure

```
energy-forecasting-xgboost/
│── notebooks/
│   └── energy_forecasting.ipynb
│── README.md
│── requirements.txt
```

pip install -r requirements.txt