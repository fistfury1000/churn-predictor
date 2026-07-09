"""
app.py

Streamlit front end for the Churn Prediction API.

This is a thin UI layer — it calls the FastAPI backend (api/main.py)
rather than loading the model directly, so the API stays the single
source of truth for predictions.

Run (with the API already running in another terminal):
    streamlit run app.py
"""

import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Risk Predictor")
st.caption(
    "XGBoost model with SHAP-based explainability, served via FastAPI. "
    "Fill in a customer's profile below to see their churn risk and the "
    "top factors driving that prediction."
)

with st.form("customer_form"):
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=1.0)

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Can't reach the API. Make sure it's running in another terminal:\n\n"
            "`uvicorn api.main:app --reload`"
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}")
        st.stop()

    st.divider()
    st.subheader("Prediction")

    proba = result["churn_probability"]
    risk = result["risk_level"]

    risk_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

    m1, m2 = st.columns(2)
    m1.metric("Churn Probability", f"{proba:.1%}")
    m2.metric("Risk Level", f"{risk_colors.get(risk, '')} {risk}")

    st.progress(min(proba, 1.0))

    st.subheader("Top Factors Driving This Prediction")

    factors = result["top_factors"]
    factors_df = pd.DataFrame(factors)[["feature", "value", "shap_value"]]
    factors_df.columns = ["Feature", "Value", "SHAP Impact"]

    st.dataframe(factors_df, use_container_width=True, hide_index=True)

    st.bar_chart(
        data=factors_df.set_index("Feature")["SHAP Impact"],
        use_container_width=True,
    )

    st.caption(
        "Positive SHAP values push the prediction toward churn; negative values push toward retention."
    )
