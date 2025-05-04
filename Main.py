import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("🔍 Fraud Detection System")
st.markdown("Enter transaction details below to classify it as **Fraudulent** or **Genuine**.")

# List of feature names excluding target 'isFraud'
features = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'transactionType',
    'deviceType', 'region', 'hour', 'dayofweek', 'txnCount24h',
    'avgAmount30d', 'isForeignTransaction', 'isHighRiskCountry',
    'hasSecureAuth'
]

# User input
user_input = []

# Numerical features
numerical_features = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'region', 'hour',
    'dayofweek', 'txnCount24h', 'avgAmount30d'
]

# Categorical/Binary features
categorical_options = {
    'transactionType': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],
    'deviceType': ['MOBILE', 'DESKTOP'],
    'isForeignTransaction': [0, 1],
    'isHighRiskCountry': [0, 1],
    'hasSecureAuth': [0, 1]
}

# Input widgets
for feature in features:
    if feature in numerical_features:
        val = st.number_input(f"{feature}", step=1.0)
        user_input.append(val)
    elif feature in categorical_options:
        val = st.selectbox(f"{feature}", categorical_options[feature])
        # Encode as integer (optional: use your model's actual encoding if needed)
        if isinstance(val, str):
            val = categorical_options[feature].index(val)
        user_input.append(val)

# Predict
if st.button("🔎 Predict Fraud"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result:")
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Transaction is Genuine (Confidence: {100 - prob*100:.2f}%)")
