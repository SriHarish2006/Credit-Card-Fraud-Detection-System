"""
train_model.py
---------------
End-to-end training script for the Credit Card Fraud Detection project.

Run directly:
    python src/train_model.py

Pipeline:
    Dataset -> Validate -> Clean -> Train/Test Split (stratified)
            -> Scale (fit on train only)
            -> SMOTE (train only)
            -> Train Logistic Regression + Random Forest
            -> Evaluate both on the untouched test set
            -> Pick the best model using a recall/F1/ROC-AUC-aware rule
            -> Save model, scaler, and evaluation report to disk
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import RANDOM_STATE, full_preprocessing_pipeline
from src.evaluate_model import evaluate_classifier, select_best_model

DATA_PATH = os.path.join("data", "creditcard.csv")
MODELS_DIR = "models"
REPORTS_DIR = os.path.join("outputs", "reports")


def apply_smote(X_train, y_train):
    """Apply SMOTE to the TRAINING split only, and report the before/after
    class distribution so the imbalance-handling step is transparent."""
    before_genuine = int((y_train == 0).sum())
    before_fraud = int((y_train == 1).sum())
    print(f"Before SMOTE -> Genuine: {before_genuine}, Fraud: {before_fraud}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    after_genuine = int((y_resampled == 0).sum())
    after_fraud = int((y_resampled == 1).sum())
    print(f"After SMOTE  -> Genuine: {after_genuine}, Fraud: {after_fraud}")

    return X_resampled, y_resampled, {
        "before": {"genuine": before_genuine, "fraud": before_fraud},
        "after": {"genuine": after_genuine, "fraud": after_fraud},
    }


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Logistic Regression baseline with balanced class weights, as an
    additional safeguard alongside SMOTE against the class imbalance."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Random Forest ensemble model, configured with reasonable defaults
    for a moderately-sized tabular dataset."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading and preprocessing dataset")
    print("=" * 60)
    pipeline_output = full_preprocessing_pipeline(DATA_PATH)

    X_train, X_test = pipeline_output["X_train"], pipeline_output["X_test"]
    y_train, y_test = pipeline_output["y_train"], pipeline_output["y_test"]
    scaler = pipeline_output["scaler"]
    feature_names = pipeline_output["feature_names"]
    raw_report = pipeline_output["raw_report"]

    print(f"Dataset shape: {raw_report.n_rows} rows x {raw_report.n_cols} columns")
    print(f"Genuine: {raw_report.genuine_count} ({raw_report.genuine_percentage}%)")
    print(f"Fraud:   {raw_report.fraud_count} ({raw_report.fraud_percentage}%)")

    print("\n" + "=" * 60)
    print("STEP 2: Handling class imbalance with SMOTE (training set only)")
    print("=" * 60)
    X_train_res, y_train_res, smote_summary = apply_smote(X_train, y_train)

    print("\n" + "=" * 60)
    print("STEP 3: Training models")
    print("=" * 60)

    t0 = time.time()
    log_reg = train_logistic_regression(X_train_res, y_train_res)
    print(f"Logistic Regression trained in {time.time() - t0:.2f}s")

    t0 = time.time()
    rand_forest = train_random_forest(X_train_res, y_train_res)
    print(f"Random Forest trained in {time.time() - t0:.2f}s")

    print("\n" + "=" * 60)
    print("STEP 4: Evaluating models on the untouched test set")
    print("=" * 60)
    results = {}
    fitted_models = {"Logistic Regression": log_reg, "Random Forest": rand_forest}

    for name, model in fitted_models.items():
        metrics = evaluate_classifier(model, X_test, y_test)
        results[name] = metrics
        print(f"\n{name}:")
        for k, v in metrics.items():
            if k not in ("confusion_matrix", "y_pred", "y_proba"):
                print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("STEP 5: Selecting the best model")
    print("=" * 60)
    best_name = select_best_model(results)
    best_model = fitted_models[best_name]
    print(f"Best model selected: {best_name}")
    print(
        "Selection rule: models are ranked primarily by fraud-class F1-score, "
        "with ROC-AUC as a tiebreaker - not by raw accuracy, since accuracy is "
        "misleading on a dataset that is ~99.8% genuine transactions."
    )

    print("\n" + "=" * 60)
    print("STEP 6: Saving model, scaler, and reports")
    print("=" * 60)

    # Save the best model + scaler + feature order together so prediction
    # code never has to guess what preprocessing was used at train time.
    joblib.dump(best_model, os.path.join(MODELS_DIR, "fraud_detection_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(
        {"feature_names": feature_names, "best_model_name": best_name},
        os.path.join(MODELS_DIR, "metadata.pkl"),
    )
    # Also persist every trained model so the app can show a full comparison.
    joblib.dump(fitted_models, os.path.join(MODELS_DIR, "all_models.pkl"))

    serializable_results = {}
    for name, metrics in results.items():
        serializable_results[name] = {
            k: v for k, v in metrics.items() if k not in ("y_pred", "y_proba")
        }

    report = {
        "dataset": {
            "n_rows": raw_report.n_rows,
            "n_cols": raw_report.n_cols,
            "genuine_count": raw_report.genuine_count,
            "fraud_count": raw_report.fraud_count,
            "fraud_percentage": raw_report.fraud_percentage,
        },
        "smote_summary": smote_summary,
        "model_results": serializable_results,
        "best_model": best_name,
    }
    with open(os.path.join(REPORTS_DIR, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel saved to: {MODELS_DIR}/fraud_detection_model.pkl")
    print(f"Scaler saved to: {MODELS_DIR}/scaler.pkl")
    print(f"Training report saved to: {REPORTS_DIR}/training_report.json")
    print("\nDone. You can now run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
