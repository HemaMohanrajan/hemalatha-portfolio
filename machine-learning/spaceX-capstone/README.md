# 🚀 SpaceX Launch Success Prediction  
**Winning the Space Race with Data Science**

Predicting Falcon 9 first-stage landing success using Machine Learning, SQL analytics, and interactive visualizations.

---

## 🎯 Project Objective

SpaceX significantly reduces launch costs by successfully recovering Falcon 9 boosters.

This project aims to:

✔ Analyze historical SpaceX launch data  
✔ Identify factors influencing landing success  
✔ Build predictive ML models  
✔ Communicate insights via dashboards & maps  

---

## 📊 Data Collection

Data was collected using:

- **SpaceX REST API**
- **Web Scraping (Wikipedia)**

Steps performed:

✔ API requests & JSON parsing  
✔ HTML table extraction using BeautifulSoup  
✔ Data normalization into Pandas DataFrames  

---

## 🧹 Data Wrangling

Performed:

- Data cleaning & structuring  
- Handling missing values  
- Feature engineering  
- Created **Landing Outcome Label**  
  - `1` → Successful landing  
  - `0` → Unsuccessful landing  

---

## 🔎 Exploratory Data Analysis (EDA)

### 📈 Key Analyses

✔ Flight Number vs Launch Site  
✔ Payload Mass vs Launch Site  
✔ Payload Mass vs Orbit Type  
✔ Success Rate vs Orbit Type  
✔ Yearly Launch Success Trend  

### 💡 Insights

- **KSC LC-39A** showed highest success rate  
- **LEO missions** had higher success probability  
- Launch success improved year-over-year  
- Payload mass influences landing outcome  

---

## 🧮 EDA with SQL

Performed SQL queries to analyze:

✔ Unique launch sites  
✔ Payload statistics (SUM / AVG)  
✔ Mission outcome grouping  
✔ Subqueries for booster performance  

---

## 🗺 Interactive Visual Analytics

### 🌍 **Folium Maps**
- Launch site markers  
- Proximity analysis (coastlines, infrastructure)  
- Success/failure visualization  

---

### 📊 **Plotly Dash Dashboard**
Features:

✔ Launch site dropdown filter  
✔ Payload mass slider  
✔ Pie charts (success/failure)  
✔ Scatterplots (payload vs success)

---

## 🤖 Predictive Modeling

### 🧠 Problem Type
Binary Classification → Landing Success

---

### 🔬 Models Evaluated

- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  
- K-Nearest Neighbors (KNN)

---

### ⚙️ Model Tuning
Used **GridSearchCV (cv=10)**

Example best parameters:

✔ Logistic Regression → L2 regularization  
✔ SVM → Sigmoid kernel  
✔ Decision Tree → Depth = 6  
✔ KNN → k = 10  

---

## 📈 Model Evaluation

| Model | Test Accuracy |
|------|--------------|
| Logistic Regression | 83.33% |
| SVM | 83.33% |
| Decision Tree | 83.33% |
| KNN | 83.33% |

✔ Confusion matrices analyzed  
✔ Decision Tree selected based on training performance

---

## ✅ Key Takeaways

✨ Multiple ML models achieved similar performance  
✨ Launch site & orbit type strongly influence outcomes  
✨ Payload mass plays a critical role  
✨ Visual analytics enhanced interpretability  

---

## 🛠 Tech Stack

**Languages**  
Python • SQL  

**Libraries**  
Pandas • NumPy • Scikit-learn  
Matplotlib • Seaborn  
Folium • Plotly Dash  
BeautifulSoup  

---

## ▶️ How to Run

1️⃣ Clone repository  

2️⃣ Install dependencies  

```bash
pip install -r requirements.txt
