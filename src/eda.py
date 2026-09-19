"""
eda.py
------
Exploratory Data Analysis helpers for the Credit Card Fraud Detection
project. Functions here compute statistics and build Plotly figures that
are reused both in the Jupyter notebook and inside the Streamlit app, so
numbers are guaranteed to be calculated dynamically from the actual data
(never hard-coded).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TARGET_COLUMN = "Class"


def fraud_statistics(df: pd.DataFrame) -> dict:
    """Return a dict of dynamically computed fraud statistics."""
    total = len(df)
    fraud = int((df[TARGET_COLUMN] == 1).sum())
    genuine = int((df[TARGET_COLUMN] == 0).sum())
    return {
        "total_transactions": total,
        "genuine_transactions": genuine,
        "fraudulent_transactions": fraud,
        "genuine_percentage": round((genuine / total) * 100, 4) if total else 0.0,
        "fraud_percentage": round((fraud / total) * 100, 4) if total else 0.0,
    }


def class_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart: genuine vs fraudulent transaction counts."""
    counts = df[TARGET_COLUMN].value_counts().rename({0: "Genuine", 1: "Fraud"})
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Transaction Type", "y": "Count"},
        title="Genuine vs Fraudulent Transactions",
        color=counts.index,
        color_discrete_map={"Genuine": "#2E7D32", "Fraud": "#C62828"},
        text=counts.values,
    )
    fig.update_layout(showlegend=False)
    return fig


def amount_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of transaction amounts (all transactions)."""
    fig = px.histogram(
        df,
        x="Amount",
        nbins=60,
        title="Transaction Amount Distribution",
        color_discrete_sequence=["#1565C0"],
    )
    fig.update_layout(xaxis_title="Amount", yaxis_title="Frequency")
    return fig


def fraud_amount_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of transaction amounts for fraudulent transactions only."""
    fraud_df = df[df[TARGET_COLUMN] == 1]
    fig = px.histogram(
        fraud_df,
        x="Amount",
        nbins=40,
        title="Fraudulent Transaction Amount Distribution",
        color_discrete_sequence=["#C62828"],
    )
    fig.update_layout(xaxis_title="Amount", yaxis_title="Frequency")
    return fig


def time_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of transaction Time (seconds since first transaction)."""
    fig = px.histogram(
        df,
        x="Time",
        color=df[TARGET_COLUMN].map({0: "Genuine", 1: "Fraud"}),
        nbins=48,
        barmode="overlay",
        opacity=0.6,
        title="Transaction Time Distribution",
        color_discrete_map={"Genuine": "#1565C0", "Fraud": "#C62828"},
    )
    fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Frequency", legend_title="Type")
    return fig


def correlation_heatmap(df: pd.DataFrame, sample_size: int = 50000) -> go.Figure:
    """Correlation heatmap of numeric features.

    Sampling is used for very large datasets purely to keep the heatmap
    computation fast in the Streamlit app; correlations are stable at
    this sample size for a dataset this size.
    """
    data = df
    if len(df) > sample_size:
        data = df.sample(sample_size, random_state=42)
    corr = data.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    return fig


def fraud_trend_over_time_chart(df: pd.DataFrame, bins: int = 48) -> go.Figure:
    """Fraud rate over time buckets, to reveal any temporal patterns."""
    df = df.copy()
    df["time_bucket"] = pd.cut(df["Time"], bins=bins)
    trend = df.groupby("time_bucket", observed=True)[TARGET_COLUMN].mean().reset_index()
    trend["time_bucket_mid"] = trend["time_bucket"].apply(lambda x: x.mid)
    fig = px.line(
        trend,
        x="time_bucket_mid",
        y=TARGET_COLUMN,
        title="Fraud Rate Trend Over Time",
        markers=True,
    )
    fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Fraud Rate")
    return fig


def dataset_summary_text(df: pd.DataFrame) -> dict:
    """Textual dataset info block used on the Dashboard / EDA page."""
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
