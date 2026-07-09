"""
train.py

Trains an XGBoost classifier on the Telco Customer Churn dataset,
evaluates it, and saves the model + encoders + feature names + a
SHAP explainer for downstream use by the API.

Run from the project root:
    python -m src.train
"""

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
import shap

from src.data_loader import load_and_prepare

MODEL_PATH = "models/churn_model.joblib"
ENCODERS_PATH = "models/encoders.joblib"
FEATURES_PATH = "models/feature_names.joblib"
EXPLAINER_PATH = "models/shap_explainer.joblib"

RANDOM_STATE = 42


def train():
    print("Loading data...")
    X, y, encoders, feature_names = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # class imbalance is mild in this dataset (~27% churn) but worth handling
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # --- Feature importance sanity check ---
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features (XGBoost gain-based importance):")
    for i in top_idx:
        print(f"  {feature_names[i]:25s} {importances[i]:.4f}")

    # --- SHAP explainer (built once, reused at inference) ---
    print("\nBuilding SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    # --- Persist everything the API needs ---
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    joblib.dump(explainer, EXPLAINER_PATH)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved encoders to {ENCODERS_PATH}")
    print(f"Saved feature names to {FEATURES_PATH}")
    print(f"Saved SHAP explainer to {EXPLAINER_PATH}")


if __name__ == "__main__":
    train()
