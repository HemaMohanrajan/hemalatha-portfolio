# 🎓 Student Dropout & Academic Success Prediction

Predicting student academic outcomes (**Dropout / Enrolled / Graduate**) using Machine Learning.

---

## 🚀 Problem Statement

Student attrition is a major challenge for higher education institutions.  
Early identification of students at risk of dropping out enables proactive academic support and intervention strategies.

This project applies **machine learning classification models** to predict student outcomes using demographic, academic, and socio-economic data available at enrollment.

---

## 📊 Dataset

- **Records:** 4,424 students  
- **Features:** 37 attributes  
- **Target Variable:**  
  - Dropout  
  - Enrolled  
  - Graduate  

Dataset includes:

✔ Demographics  
✔ Academic performance  
✔ Socio-economic indicators  
✔ Macro-economic variables

---

## 🔎 Exploratory Data Analysis (EDA)

Performed:

- Data overview & profiling
- Target distribution analysis
- Outlier detection (Z-score)
- Correlation analysis
- Feature selection / reduction

### 🎯 Target Distribution

Graduate: 2209  
Dropout: 1421  
Enrolled: 794  

*(Class imbalance considered during modeling)*

---

## 🧹 Data Preparation

✔ Checked missing values → None  
✔ Checked duplicates → None  
✔ Identified numerical & categorical features  
✔ Removed low-signal / redundant features

Feature reduction improved model efficiency and interpretability.

---

## 🤖 Modeling Approach

This is a **multi-class classification problem**.

Models evaluated:

- Logistic Regression  
- Decision Tree  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)

---

## 📈 Evaluation Metrics

Used:

✔ Accuracy  
✔ Precision  
✔ Recall  
✔ F1-score  
✔ Confusion Matrix  

---

## ✅ Key Insights

- Academic performance features strongly influence outcomes
- Certain demographic & socio-economic factors correlate with dropout risk
- Feature reduction improved model stability

---

## 🛠 Tech Stack

Python • Pandas • NumPy  
Scikit-learn  
Matplotlib • Seaborn  

---

## ▶️ How to Run

1️⃣ Clone repository  

2️⃣ Install dependencies  

```bash
pip install -r requirements.txt
