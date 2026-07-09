"""
data_loader.py

Loads and preprocesses the Telco Customer Churn dataset.

Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Expected file: data/telco_churn.csv
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/telco_churn.csv"

# Columns we drop outright — customerID is an identifier, not a feature
DROP_COLS = ["customerID"]

TARGET_COL = "Churn"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - TotalCharges is read as object due to blank strings for new customers
      (tenure == 0). Coerce to numeric, fill blanks with 0.
    - Drop identifier columns.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    return df


def encode_features(df: pd.DataFrame, encoders: dict = None):
    """
    Label-encode all categorical (object) columns except the target.
    If `encoders` dict is provided, reuse them (inference time).
    Otherwise fit new encoders (training time) and return them.

    Returns:
        df_encoded, encoders_dict
    """
    df = df.copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != TARGET_COL]

    for col in cat_cols:
        if fit_mode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # handle unseen categories gracefully at inference time
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df, encoders


def load_and_prepare(path: str = RAW_PATH):
    """
    Full pipeline: load -> clean -> encode -> split X/y.

    Returns:
        X (DataFrame), y (Series, 1=Churn/Yes, 0=No), encoders (dict), feature_names (list)
    """
    df = load_raw(path)
    df = clean(df)

    y = df[TARGET_COL].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[TARGET_COL])

    X_encoded, encoders = encode_features(X)

    return X_encoded, y, encoders, list(X_encoded.columns)


if __name__ == "__main__":
    X, y, encoders, feature_names = load_and_prepare()
    print(f"Loaded {len(X)} rows, {len(feature_names)} features")
    print(f"Churn rate: {y.mean():.2%}")
    print(f"Features: {feature_names}")
