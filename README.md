# Customer Churn Prediction

Predicting telecom customer churn using machine learning, with end-to-end analysis from EDA to business ROI.

**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Author:** Papimon Kongnark

---

## Overview

This project builds a churn prediction system for a telecom company using the IBM Telco dataset (7,032 customers, 26.6% churn rate). The pipeline covers exploratory data analysis, feature engineering, model training with class imbalance handling (SMOTE), threshold optimization, and business ROI quantification.

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── Telco_Customer_Churn.csv
├── notebooks/
│   └── customer_churn.ipynb     # Full pipeline: EDA → modeling → business impact
├── requirements.txt
└── README.md
```

---

## Pipeline

| Step | Description |
|------|-------------|
| 1. EDA | Churn distribution, numeric/categorical breakdowns, correlation heatmap |
| 2. Feature Engineering | Binary encoding, one-hot encoding, StandardScaler, SMOTE for class imbalance |
| 3. Model Training | Logistic Regression, Random Forest, XGBoost, Gradient Boosting |
| 4. Evaluation | ROC-AUC, Precision-Recall, threshold sensitivity analysis |
| 5. Business Impact | ROI calculation, net benefit, waterfall chart |

---

## Model Results

All models trained on SMOTE-resampled data (train/test split 80/20, stratified).

| Model | CV AUC | Test AUC | Avg Precision |
|-------|--------|----------|---------------|
| Logistic Regression | 0.8453 | 0.8341 | 0.6236 |
| Random Forest | 0.8492 | 0.8340 | 0.6332 |
| XGBoost | 0.8415 | 0.8254 | 0.6297 |
| **Gradient Boosting** | **0.8454** | **0.8353** | **0.6430** |

**Best model: Gradient Boosting** (Test AUC = 0.8353)

### Classification Report (Gradient Boosting, threshold = 0.50)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Churn | 0.87 | 0.82 | 0.85 | 1,033 |
| Churn | 0.58 | 0.67 | 0.62 | 374 |
| **Accuracy** | | | **0.78** | **1,407** |

---

## Business Impact

Assumptions: ARPU = $65/month, retention period = 24 months, retention cost = $50/customer.

| Metric | Value |
|--------|-------|
| Optimal threshold | 0.100 |
| Customers correctly identified as at-risk | 267 / 374 |
| Net savings vs. doing nothing | **$222,400** |
| ROI on retention investment | **71.4%** |
| Cost of False Positives (wasted spend) | $44,200 |
| Cost of False Negatives (missed churners) | $214,000 |
| Profit gain from threshold tuning (0.5 → 0.1) | +$27,600 |

> Lowering the decision threshold from 0.5 to 0.1 significantly increases recall for at-risk customers, reducing missed revenue at the expense of a modest increase in wasted retention spend — the net effect is a $27,600 improvement in profitability.

---

## Key EDA Findings

| Feature | Insight |
|---------|---------|
| Contract Type | Month-to-month customers churn at **42.7%** — highest of any contract type |
| Internet Service | Fiber optic users churn at **41.9%** |
| Payment Method | Electronic check correlates with **45.3%** churn rate |
| Senior Citizens | Churn at **41.7%** vs 23.6% for non-seniors |
| Tenure | Customers in their first 12 months have the highest churn risk |
| Tech Support / Online Security | Absence of these services strongly correlates with churn |

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `joblib`

Open `notebooks/customer_churn.ipynb` to run the full pipeline.

---

## Next Steps

- Hyperparameter tuning with Optuna
- Stacking / ensemble models
- REST API deployment with FastAPI
- Business dashboard with Streamlit
