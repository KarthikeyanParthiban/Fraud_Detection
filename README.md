# Fraud Detection with Synthentic Dataset

## Overview

This project demonstrates how to build a synthetic fraud detection system using a combination of data simulation, preprocessing, and machine learning classification models. The project includes feature engineering, class balancing with SMOTE, model benchmarking with LazyPredict, and final model training and evaluation using XGBoost.

---

## 📌 Scope

* Simulate a fraud transaction dataset with realistic features (amounts, balances, transaction type, region, device type, etc.).
* Address class imbalance (only 2.5% fraud) using **SMOTE**.
* Benchmark models using **LazyPredict**.
* Finalize and evaluate the model using **XGBoost (XGBClassifier)**.
* Final model training using XGBoost.
* Deployment of a fraud prediction web app using Streamlit.

---

## 🔧 Requirements

* Python 3.8+
* pandas, numpy
* sklearn
* faker
* seaborn, matplotlib
* lazypredict
* imbalanced-learn
* xgboost
* streamlit, joblib

Install all dependencies using:

```bash
pip install pandas numpy scikit-learn faker matplotlib seaborn lazypredict imbalanced-learn xgboost streamlit joblib
```

---

## 🧪 Dataset Generation

* 10,000 samples simulated using `make_classification`.
* Fraud ratio: **2.5%**.
* Realistic engineered features:

  * Transaction amount, old/new balances, device, region
  * Time-based features (hour, day of week)
  * Behavioral features (transaction count in 24h, average amount in 30d)
  * Security features (is foreign transaction, is high risk country)

---

## 📊 Exploratory Data Analysis

* Visualized class imbalance.
* Histogram distribution of numerical variables grouped by `isFraud`.
* Fraud by hour of the day.
* Feature importances using a Random Forest model.

---

## ⚙️ Preprocessing

* Label encoding for categorical variables.
* Scaling using `StandardScaler`.
* Balanced data using `SMOTE` oversampling.

---

## 🚀 Modeling and Evaluation

### Benchmarking with LazyPredict

Top 10 models evaluated using LazyPredict after resampling:

| Model                  | Accuracy | ROC AUC | F1 Score |
| ---------------------- | -------- | ------- | -------- |
| XGBClassifier          | 91%      | 0.51    | 92%      |
| ExtraTreeClassifier    | 82%      | 0.51    | 88%      |
| RandomForestClassifier | 93%      | 0.49    | 94%      |
| LogisticRegression     | 58%      | 0.49    | 71%      |

> Many models showed inflated accuracy due to class imbalance. After using SMOTE, the dataset was balanced and re-evaluated.

### Final Model: XGBoost Classifier

`XGBClassifier` from the XGBoost library was used due to its robust performance and ability to handle imbalanced datasets effectively.

**Model Highlights:**

* **Accuracy**: 91%
* **Balanced Accuracy**: 51%
* **ROC AUC**: 51%
* **F1 Score**: 92%
* **Time Taken**: 0.09 seconds

Despite strong F1 and accuracy scores, low ROC AUC suggests more work is needed to optimize true fraud detection.

---

## 💾 Model Saving

Model and scaler are saved for future deployment:

```python
import joblib
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

## ✅ Results Summary

* Synthetic fraud detection system built end-to-end.
* Balanced class using SMOTE.
* Top model: XGBoost.
* Ready for further tuning and deployment.

---

## 🌐 Deployment with Streamlit

---
* Clean UI for inputting transaction details

* Categorical and numerical feature support

* Shows prediction with confidence score

---

 **Sample UI Inputs**

 ---

| Feature              | Type        | Description                     |
| -------------------- | ----------- | ------------------------------- |
| amount               | Numeric     | Transaction amount              |
| transactionType      | Categorical | PAYMENT, TRANSFER, etc.         |
| deviceType           | Categorical | MOBILE, DESKTOP                 |
| region               | Numeric     | Region code (e.g., 1–10)        |
| hour, dayofweek      | Numeric     | Time-related transaction info   |
| isForeignTransaction | Binary      | 1 if international, else 0      |
| isHighRiskCountry    | Binary      | 1 if risky destination, else 0  |
| hasSecureAuth        | Binary      | 1 if secure authentication used |

---

## 📁 Repository Structure

```
fraud_detection_project/
├── fraud_detection.ipynb    # Notebook for full analysis
├── app.py                   # Streamlit app for deployment
├── fraud_model.pkl          # Trained XGBoost model
├── scaler.pkl               # StandardScaler object
└── README.md                # Project documentation

```

## 🧠 Author & License

Built by \Karthikeyan-parthiban.
