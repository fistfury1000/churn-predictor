"""
explain.py

Turns raw SHAP values for a single prediction into a human-readable
list of the top factors driving that prediction, e.g.:

    [
        {"feature": "Contract", "value": "Month-to-month", "impact": "+0.31 (increases risk)"},
        {"feature": "tenure", "value": 2, "impact": "+0.18 (increases risk)"},
        {"feature": "TechSupport", "value": "No", "impact": "+0.12 (increases risk)"},
    ]
"""

import pandas as pd


def top_factors(explainer, X_row: pd.DataFrame, raw_row: pd.Series, top_n: int = 3):
    """
    Args:
        explainer: fitted shap.TreeExplainer
        X_row: single-row encoded DataFrame (model input format)
        raw_row: single-row original (human-readable) values, same index as X_row columns
        top_n: how many top factors to return

    Returns:
        List of dicts describing the top_n most influential features for this prediction.
    """
    shap_values = explainer.shap_values(X_row)

    # shap_values shape: (1, n_features)
    values = shap_values[0]
    feature_names = X_row.columns.tolist()

    # pair up (feature, shap_value), sort by absolute impact
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)

    results = []
    for feature, shap_val in pairs[:top_n]:
        direction = "increases risk" if shap_val > 0 else "decreases risk"
        raw_val = raw_row[feature] if feature in raw_row else X_row[feature].values[0]
        results.append({
            "feature": feature,
            "value": raw_val,
            "shap_value": round(float(shap_val), 4),
            "impact": f"{shap_val:+.3f} ({direction})",
        })

    return results
