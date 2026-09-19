"""
evaluate_model.py
------------------
Evaluation utilities shared by train_model.py and the Streamlit app.

We deliberately do NOT judge models by accuracy alone. With fraud making
up roughly 0.17% of transactions, a model that predicts "genuine" for
every transaction would already score ~99.8% accuracy while catching
zero fraud. Precision, recall, F1-score (for the fraud class) and
ROC-AUC give a much more honest picture.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def evaluate_classifier(model, X_test, y_test) -> Dict:
    """Compute the full evaluation-metric suite for a fitted classifier."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:  # pragma: no cover - both our models support predict_proba
        y_proba = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return metrics


def get_roc_curve_data(y_test, y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    return fpr, tpr, thresholds


def get_precision_recall_curve_data(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    return precision, recall, thresholds


def select_best_model(results: Dict[str, Dict]) -> str:
    """Pick the best model using a fraud-aware ranking rule.

    Primary key : fraud-class F1-score (balances precision and recall)
    Tiebreaker  : ROC-AUC
    Accuracy is intentionally excluded from the ranking because it is
    misleading on an imbalanced dataset like this one.
    """
    ranked = sorted(
        results.items(),
        key=lambda item: (item[1]["f1_score"], item[1]["roc_auc"]),
        reverse=True,
    )
    return ranked[0][0]


def feature_importance(model, feature_names):
    """Return a sorted list of (feature, importance) tuples if the model
    supports it (e.g. RandomForest); otherwise return None gracefully."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs
