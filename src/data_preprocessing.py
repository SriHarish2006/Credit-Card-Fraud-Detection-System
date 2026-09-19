"""
data_preprocessing.py
----------------------
Handles dataset loading, validation, cleaning, feature/target separation,
scaling, and train/test splitting for the Credit Card Fraud Detection project.

IMPORTANT (avoiding data leakage):
    SMOTE (class-imbalance oversampling) is intentionally NOT performed in
    this module. It must only ever be applied to the *training* split,
    which is done separately in `train_model.py` after `train_test_split`.
    Applying SMOTE before splitting would leak synthetic copies of the same
    fraud pattern into both train and test sets, giving an unrealistically
    optimistic (and incorrect) evaluation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REQUIRED_V_COLUMNS = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = ["Time"] + REQUIRED_V_COLUMNS + ["Amount", "Class"]
TARGET_COLUMN = "Class"
RANDOM_STATE = 42


class DatasetValidationError(Exception):
    """Raised when the dataset does not match the expected schema."""


@dataclass
class DatasetReport:
    """Small summary object describing the raw dataset, useful for the
    Streamlit dashboard and for printing during training."""

    n_rows: int
    n_cols: int
    columns: list
    missing_values: dict
    duplicate_rows: int
    dtypes: dict
    genuine_count: int
    fraud_count: int
    fraud_percentage: float
    genuine_percentage: float


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the credit card dataset from disk with a clear error if missing."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Please place the creditcard.csv file inside the data/ folder."
        )
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise DatasetValidationError(f"Failed to read CSV file: {exc}") from exc
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate that the dataset has the columns this project expects.

    Raises a DatasetValidationError with a clear, human-readable message
    instead of letting the pipeline crash with a cryptic KeyError later.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetValidationError(
            "The dataset is missing required columns: "
            f"{missing_cols}. Expected columns are Time, V1-V28, Amount, Class."
        )

    if df.empty:
        raise DatasetValidationError("The dataset file was loaded but contains no rows.")

    unique_targets = set(pd.unique(df[TARGET_COLUMN].dropna()))
    if not unique_targets.issubset({0, 1, "0", "1"}):
        raise DatasetValidationError(
            f"Target column '{TARGET_COLUMN}' must only contain 0 (genuine) "
            f"and 1 (fraud). Found values: {unique_targets}"
        )


def profile_dataset(df: pd.DataFrame) -> DatasetReport:
    """Build a DatasetReport summarising shape, quality, and class balance."""
    class_counts = df[TARGET_COLUMN].astype(int).value_counts()
    genuine_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    total = genuine_count + fraud_count

    return DatasetReport(
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        columns=list(df.columns),
        missing_values=df.isnull().sum().to_dict(),
        duplicate_rows=int(df.duplicated().sum()),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        genuine_count=genuine_count,
        fraud_count=fraud_count,
        fraud_percentage=round((fraud_count / total) * 100, 4) if total else 0.0,
        genuine_percentage=round((genuine_count / total) * 100, 4) if total else 0.0,
    )


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Clean the dataset: handle missing values and duplicate rows.

    The public creditcard.csv dataset is usually free of missing values,
    but we handle them defensively in case a different/partial file is
    supplied. Numeric columns are imputed with the median (robust to the
    heavy skew typical of fraud data); rows missing the target are dropped
    since they cannot be used for supervised learning.
    """
    df = df.copy()

    # Drop rows with a missing target - they can't be used for training.
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Impute missing numeric feature values with the median.
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    for col in feature_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            print(f"[data_preprocessing] Removed {removed} duplicate rows.")

    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from the target (y)."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def train_test_split_stratified(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Stratification is essential here: with fraud making up roughly 0.17%
    of transactions, a plain random split could easily leave the test set
    with very few (or disproportionate) fraud examples, making evaluation
    unreliable.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the TRAINING data only (never on test data)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def scale_features(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    """Apply an already-fitted scaler to a feature matrix."""
    return scaler.transform(X)


def full_preprocessing_pipeline(
    csv_path: str, test_size: float = 0.2
):
    """Convenience wrapper running load -> validate -> clean -> split -> scale.

    Returns a dict with everything downstream code (training, prediction)
    needs, so training and inference always use identical preprocessing.
    """
    df = load_dataset(csv_path)
    validate_dataset(df)
    report = profile_dataset(df)
    df_clean = clean_dataset(df)

    X, y = split_features_target(df_clean)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size)

    scaler = fit_scaler(X_train)
    X_train_scaled = scale_features(scaler, X_train)
    X_test_scaled = scale_features(scaler, X_test)

    return {
        "raw_report": report,
        "feature_names": list(X.columns),
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "scaler": scaler,
    }
