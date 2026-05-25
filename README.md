# Customer Churn Prediction

> End-to-end machine learning system for telecom customer churn — from exploratory analysis to business ROI, with an interactive Streamlit demo.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-red)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-ff4b4b?logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-blueviolet)

**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Author:** Papimon Kongnark

---

## Overview

This project builds a production-ready churn prediction system for a telecom company using the IBM Telco dataset (7,032 customers, 26.6% churn rate).

The pipeline covers:
- Exploratory data analysis with business-relevant insights
- Feature engineering and class imbalance handling (SMOTE)
- Multi-model comparison with threshold optimization
- Business ROI quantification ($222,400 net savings)
- SHAP explainability for stakeholder transparency
- Interactive Streamlit demo for real-time prediction

---

## Live Demo

Run the Streamlit app locally to predict churn for individual customers in real time:

```bash
streamlit run streamlit/app.py
```

Input customer tenure, monthly charges, and contract type — the model returns a churn probability with a visual risk indicator.

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── Telco_Customer_Churn.csv        # IBM Telco dataset (7,032 rows)
├── notebooks/
│   ├── customer_churn.ipynb            # Full pipeline: EDA → modeling → business impact
│   ├── best_churn_model.pkl            # Saved Gradient Boosting model
│   └── scaler.pkl                      # Fitted StandardScaler
├── streamlit/
│   └── app.py                          # Interactive churn prediction demo
├── requirements.txt
└── README.md
```

---

## ML Pipeline

| Step | Description |
|------|-------------|
| 1. EDA | Churn distribution, numeric/categorical breakdowns, correlation heatmap |
| 2. Feature Engineering | Binary encoding, one-hot encoding, StandardScaler, SMOTE for class imbalance |
| 3. Model Training | Logistic Regression, Random Forest, XGBoost, Gradient Boosting |
| 4. Evaluation | ROC-AUC, Precision-Recall, threshold sensitivity analysis |
| 5. Business Impact | ROI calculation, net benefit, waterfall chart |
| 6. SHAP Explainability | Global feature importance (bar, beeswarm), local waterfall, dependence plots |

---

## Model Results

All models trained on SMOTE-resampled data (80/20 stratified train/test split).

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
| Optimal threshold | 0.10 |
| Customers correctly identified as at-risk | 267 / 374 |
| Net savings vs. doing nothing | **$222,400** |
| ROI on retention investment | **71.4%** |
| Cost of False Positives (wasted spend) | $44,200 |
| Cost of False Negatives (missed churners) | $214,000 |
| Profit gain from threshold tuning (0.5 → 0.1) | **+$27,600** |

> Lowering the decision threshold from 0.5 to 0.1 increases recall for at-risk customers significantly. The net effect is a **$27,600 improvement in profitability** by catching more churners at the cost of modest additional retention spend.

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
# Clone the repo
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit demo
streamlit run streamlit/app.py
```

Open `notebooks/customer_churn.ipynb` to run the full training pipeline.

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `joblib`, `streamlit`

---

<img width="1366" height="641" alt="image" src="https://github.com/user-attachments/assets/06953668-3a34-4413-92c0-4d5f45cf9c54" />

## Next Steps

- [ ] Hyperparameter tuning with Optuna
- [ ] Stacking / ensemble models
- [ ] REST API deployment with FastAPI
- [ ] Full-featured Streamlit dashboard with SHAP explanations per prediction
