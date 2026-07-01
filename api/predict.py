"""
GraphShield AI
Prediction Service
"""

from pathlib import Path

import joblib
import pandas as pd

# ------------------------------------------------------
# Load trained XGBoost model
# ------------------------------------------------------

MODEL_PATH = Path("outputs/models/xgboost.pkl")

model = joblib.load(MODEL_PATH)


def predict_transaction(transaction):

    # Convert request to DataFrame
    df = pd.DataFrame([transaction])

    # Predict probability
    probability = model.predict_proba(df)[0][1]

    prediction = int(probability >= 0.50)

    # Risk level
    if probability >= 0.80:
        risk = "High"
    elif probability >= 0.50:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "prediction": prediction,
        "fraud_probability": round(float(probability), 4),
        "risk_level": risk
    }