"""
main.py

FastAPI service that serves churn risk predictions with SHAP-based
explanations for each prediction.

Run from the project root:
    uvicorn api.main:app --reload

Then POST to http://127.0.0.1:8000/predict
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explain import top_factors

MODEL_PATH = "models/churn_model.joblib"
ENCODERS_PATH = "models/encoders.joblib"
FEATURES_PATH = "models/feature_names.joblib"
EXPLAINER_PATH = "models/shap_explainer.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn risk with SHAP-based explainability.",
    version="1.0.0",
)

# --- Load model artifacts once at startup ---
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
except FileNotFoundError:
    model = encoders = feature_names = explainer = None


class CustomerFeatures(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=5)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=85.5)
    TotalCharges: float = Field(..., example=427.5)


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str
    top_factors: list


@app.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "message": "POST customer features to /predict",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` first to generate model artifacts.",
        )

    raw = pd.DataFrame([customer.dict()])

    # encode categoricals using the same encoders fit at training time
    encoded = raw.copy()
    for col, le in encoders.items():
        if col in encoded.columns:
            encoded[col] = encoded[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # ensure column order matches training
    encoded = encoded[feature_names]

    proba = float(model.predict_proba(encoded)[0, 1])

    if proba >= 0.7:
        risk_level = "High"
    elif proba >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    factors = top_factors(explainer, encoded, raw.iloc[0], top_n=3)

    return PredictionResponse(
        churn_probability=round(proba, 4),
        risk_level=risk_level,
        top_factors=factors,
    )
