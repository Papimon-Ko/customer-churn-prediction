import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "notebooks"
model  = joblib.load(BASE_DIR / "best_churn_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

st.title("Customer Churn Prediction")
st.write(
    "This application uses a trained machine learning model to predict whether "
    "a customer is likely to churn based on their demographic and usage data."
)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Customer Data Input")
tenure          = st.sidebar.slider("Customer Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 120.0, 70.0)
contract_type   = st.sidebar.selectbox(
    "Contract Type", ["Month-to-Month", "One year", "Two year"]
)

# ── Build feature vector (26 features — customer_churn.ipynb) ─────────────────
# Index  0: gender                                   (0=Male, 1=Female)
# Index  1: SeniorCitizen                            (0/1)
# Index  2: Partner                                  (0/1)
# Index  3: Dependents                               (0/1)
# Index  4: tenure                                   ← user input
# Index  5: PhoneService                             (0/1)
# Index  6: MultipleLines                            (0/1)
# Index  7: OnlineSecurity                           (0/1)
# Index  8: OnlineBackup                             (0/1)
# Index  9: DeviceProtection                         (0/1)
# Index 10: TechSupport                              (0/1)
# Index 11: StreamingTV                              (0/1)
# Index 12: StreamingMovies                          (0/1)
# Index 13: PaperlessBilling                         (0/1)
# Index 14: MonthlyCharges                           ← user input
# Index 15: TotalCharges                             ← estimated
# Index 16: InternetService_DSL                      (0/1)
# Index 17: InternetService_Fiber optic              (0/1)
# Index 18: InternetService_No                       (0/1)
# Index 19: Contract_Month-to-month                  (0/1)  ← user input
# Index 20: Contract_One year                        (0/1)  ← user input
# Index 21: Contract_Two year                        (0/1)  ← user input
# Index 22: PaymentMethod_Bank transfer (automatic)  (0/1)
# Index 23: PaymentMethod_Credit card (automatic)    (0/1)
# Index 24: PaymentMethod_Electronic check           (0/1)
# Index 25: PaymentMethod_Mailed check               (0/1)

features = np.zeros(26)

features[4]  = tenure
features[14] = monthly_charges
features[15] = tenure * monthly_charges   # TotalCharges estimate
features[18] = 1                          # InternetService_No (default)
features[25] = 1                          # PaymentMethod_Mailed check (default)

contract_idx = {"Month-to-Month": 19, "One year": 20, "Two year": 21}
features[contract_idx[contract_type]] = 1

# ── Scale → predict ───────────────────────────────────────────────────────────
features_df     = pd.DataFrame(features.reshape(1, -1), columns=scaler.feature_names_in_)
features_scaled = scaler.transform(features_df)
prediction      = model.predict(features_scaled)[0]
probability     = model.predict_proba(features_scaled)[0][1]

# ── Display inputs ────────────────────────────────────────────────────────────
st.subheader("Input Summary")
st.write(f"- **Tenure:** {tenure} months")
st.write(f"- **Monthly Charges:** ${monthly_charges:.2f}")
st.write(f"- **Contract Type:** {contract_type}")

# ── Display result ────────────────────────────────────────────────────────────
st.subheader("Prediction")
if prediction == 1:
    st.error(f"⚠️ High churn risk — {probability:.1%} probability of churning")
else:
    st.success(f"✅ Low churn risk — {probability:.1%} probability of churning")

st.progress(float(probability))
