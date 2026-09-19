"""
prediction.py
--------------
Loads the saved model + scaler and runs predictions on new transactions,
using the exact same preprocessing pipeline that was used at training
time (to avoid train/serve skew).
"""

from __future__ import annotations

import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_detection_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.pkl")
ALL_MODELS_PATH = os.path.join(MODELS_DIR, "all_models.pkl")


class ModelNotFoundError(Exception):
    """Raised when the trained model/scaler files are missing."""


def artifacts_exist() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(METADATA_PATH)


def load_artifacts():
    """Load the trained model, scaler, and metadata (feature order, model name)."""
    if not artifacts_exist():
        raise ModelNotFoundError(
            "Trained model files were not found in the 'models/' folder. "
            "Please run 'python src/train_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, scaler, metadata


def load_all_models():
    """Load the dict of every trained model (used for the comparison page)."""
    if not os.path.exists(ALL_MODELS_PATH):
        return None
    return joblib.load(ALL_MODELS_PATH)


def build_input_dataframe(transaction: Dict[str, float], feature_names: list) -> pd.DataFrame:
    """Build a single-row DataFrame from a dict of user-entered values,
    ensuring columns are in the exact order the scaler/model expect."""
    missing = [f for f in feature_names if f not in transaction]
    if missing:
        raise ValueError(f"Missing input values for: {missing}")

    row = {f: transaction[f] for f in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def predict_transaction(transaction: Dict[str, float]) -> Dict:
    """Run the full predict pipeline for a single transaction dict and
    return a clear, UI-friendly result."""
    model, scaler, metadata = load_artifacts()
    feature_names = metadata["feature_names"]

    input_df = build_input_dataframe(transaction, feature_names)

    # Basic sanity checks so obviously invalid input doesn't silently
    # produce a meaningless prediction.
    if input_df.isnull().any().any():
        raise ValueError("Input contains missing/invalid (NaN) values.")
    if not np.isfinite(input_df.values).all():
        raise ValueError("Input contains non-finite values (inf/-inf).")

    scaled_input = scaler.transform(input_df)

    prediction = int(model.predict(scaled_input)[0])
    proba = model.predict_proba(scaled_input)[0]
    genuine_proba = float(proba[0])
    fraud_proba = float(proba[1])

    return {
        "prediction": "Fraudulent" if prediction == 1 else "Genuine",
        "is_fraud": bool(prediction == 1),
        "genuine_probability": round(genuine_proba * 100, 2),
        "fraud_probability": round(fraud_proba * 100, 2),
        "model_used": metadata.get("best_model_name", "Unknown"),
    }
