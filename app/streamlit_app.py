"""
streamlit_app.py
-----------------
Credit Card Fraud Detection - Streamlit Web Application.

Run with:
    streamlit run app/streamlit_app.py

Pages:
    Dashboard, Fraud Prediction, Analytics, Model Performance
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import eda
from src.data_preprocessing import (
    DatasetValidationError,
    clean_dataset,
    load_dataset,
    profile_dataset,
    validate_dataset,
)
from src.evaluate_model import feature_importance, get_roc_curve_data, get_precision_recall_curve_data
from src.prediction import (
    ModelNotFoundError,
    artifacts_exist,
    load_all_models,
    load_artifacts,
    predict_transaction,
)

DATA_PATH = os.path.join("data", "creditcard.csv")
REPORT_PATH = os.path.join("outputs", "reports", "training_report.json")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset():
    df = load_dataset(DATA_PATH)
    validate_dataset(df)
    df = clean_dataset(df)
    return df


@st.cache_resource(show_spinner=False)
def get_model_artifacts():
    return load_artifacts()


@st.cache_resource(show_spinner=False)
def get_all_models():
    return load_all_models()


def dataset_available() -> bool:
    return os.path.exists(DATA_PATH)


def model_available() -> bool:
    return artifacts_exist()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("💳 Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "🔍 Fraud Prediction", "📊 Analytics", "🤖 Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Educational ML project. Not a production banking fraud system. "
    "No real card data should ever be entered here."
)

# ---------------------------------------------------------------------------
# Guard: dataset / model presence
# ---------------------------------------------------------------------------
if not dataset_available():
    st.error(
        "⚠️ Dataset not found. Please place `creditcard.csv` inside the `data/` "
        "folder, then reload the app."
    )
    st.stop()

try:
    df = get_dataset()
except DatasetValidationError as e:
    st.error(f"⚠️ Dataset validation failed: {e}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

stats = eda.fraud_statistics(df)

# ===========================================================================
# PAGE 1 - DASHBOARD
# ===========================================================================
if page == "🏠 Dashboard":
    st.title("💳 Credit Card Fraud Detection")
    st.caption("ML-powered dashboard summarising transactions and model health.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{stats['total_transactions']:,}")
    col2.metric("Genuine Transactions", f"{stats['genuine_transactions']:,}")
    col3.metric("Fraudulent Transactions", f"{stats['fraudulent_transactions']:,}")

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Fraud Rate", f"{stats['fraud_percentage']}%")

    if model_available():
        try:
            model, scaler, metadata = get_model_artifacts()
            from src.data_preprocessing import full_preprocessing_pipeline
            from src.evaluate_model import evaluate_classifier

            with st.spinner("Evaluating current model on a fresh test split..."):
                pipeline_output = full_preprocessing_pipeline(DATA_PATH)
                metrics = evaluate_classifier(
                    model, pipeline_output["X_test"], pipeline_output["y_test"]
                )
            col5.metric("Recall (Fraud)", f"{metrics['recall'] * 100:.2f}%")
            col6.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            col7.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        except Exception:
            col5.metric("Recall (Fraud)", "—")
            col6.metric("F1 Score", "—")
            col7.metric("ROC-AUC", "—")
    else:
        col5.metric("Recall (Fraud)", "—")
        col6.metric("F1 Score", "—")
        col7.metric("ROC-AUC", "—")
        st.info(
            "ℹ️ No trained model found yet. Run `python src/train_model.py` "
            "to train and enable predictions/metrics."
        )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eda.class_distribution_chart(df), use_container_width=True)
    with c2:
        st.plotly_chart(eda.amount_distribution_chart(df), use_container_width=True)

    st.plotly_chart(eda.fraud_amount_distribution_chart(df), use_container_width=True)

    with st.expander("📋 Dataset Quality Report"):
        summary = eda.dataset_summary_text(df)
        st.write(f"**Rows:** {summary['n_rows']:,}  |  **Columns:** {summary['n_cols']}")
        st.write(f"**Missing values (total):** {summary['missing_values_total']}")
        st.write(f"**Duplicate rows remaining:** {summary['duplicate_rows']}")

# ===========================================================================
# PAGE 2 - FRAUD PREDICTION
# ===========================================================================
elif page == "🔍 Fraud Prediction":
    st.title("🔍 Fraud Prediction")
    st.caption(
        "Enter transaction details to get a model prediction. "
        "V1-V28 are PCA-anonymised features from the original dataset; "
        "in a real deployment these would be computed automatically, not typed by hand."
    )

    if not model_available():
        st.warning(
            "⚠️ No trained model found. Please run `python src/train_model.py` "
            "before using the prediction page."
        )
        st.stop()

    model, scaler, metadata = get_model_artifacts()
    feature_names = metadata["feature_names"]

    st.info(
        "💡 Tip: Use **'Load Random Sample'** to auto-fill realistic values "
        "from the dataset if you don't have specific numbers to test with."
    )

    if "sample_values" not in st.session_state:
        st.session_state.sample_values = {f: 0.0 for f in feature_names}

    if st.button("🎲 Load Random Sample from Dataset"):
        sample_row = df.sample(1).iloc[0]
        st.session_state.sample_values = {f: float(sample_row[f]) for f in feature_names}

    with st.form("prediction_form"):
        st.subheader("Transaction Details")
        values = {}

        values["Time"] = st.number_input(
            "Time (seconds since first transaction)",
            value=float(st.session_state.sample_values.get("Time", 0.0)),
            step=1.0,
        )

        st.markdown("**Anonymised PCA Features (V1 - V28)**")
        v_cols = st.columns(4)
        for i in range(1, 29):
            fname = f"V{i}"
            with v_cols[(i - 1) % 4]:
                values[fname] = st.number_input(
                    fname,
                    value=float(st.session_state.sample_values.get(fname, 0.0)),
                    format="%.6f",
                    key=f"input_{fname}",
                )

        values["Amount"] = st.number_input(
            "Amount",
            min_value=0.0,
            value=float(st.session_state.sample_values.get("Amount", 0.0)),
            step=1.0,
        )

        submitted = st.form_submit_button("🔍 Predict Transaction")

    if submitted:
        try:
            result = predict_transaction(values)
        except ValueError as e:
            st.error(f"⚠️ Invalid input: {e}")
        except ModelNotFoundError as e:
            st.error(f"⚠️ {e}")
        except Exception as e:
            st.error("⚠️ An unexpected error occurred while generating the prediction.")
            st.caption(f"Details: {type(e).__name__}")
        else:
            st.markdown("---")
            if result["is_fraud"]:
                st.error("🚨 **FRAUDULENT TRANSACTION**")
            else:
                st.success("✅ **GENUINE TRANSACTION**")

            r1, r2 = st.columns(2)
            r1.metric("Prediction", result["prediction"])
            r2.metric("Model Used", result["model_used"])

            p1, p2 = st.columns(2)
            p1.metric("Genuine Probability", f"{result['genuine_probability']}%")
            p2.metric("Fraud Probability", f"{result['fraud_probability']}%")

            fig = go.Figure(
                go.Bar(
                    x=["Genuine", "Fraud"],
                    y=[result["genuine_probability"], result["fraud_probability"]],
                    marker_color=["#2E7D32", "#C62828"],
                    text=[f"{result['genuine_probability']}%", f"{result['fraud_probability']}%"],
                    textposition="auto",
                )
            )
            fig.update_layout(title="Predicted Probability", yaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "ℹ️ This is the model's predicted probability based on patterns learned "
                "from historical data — not a guaranteed real-world risk score."
            )

# ===========================================================================
# PAGE 3 - ANALYTICS
# ===========================================================================
elif page == "📊 Analytics":
    st.title("📊 Analytics")
    st.caption("Deeper exploratory analysis of the transaction dataset.")

    st.subheader("Transaction Analysis")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Transactions", f"{stats['total_transactions']:,}")
    a2.metric("Genuine Transactions", f"{stats['genuine_transactions']:,}")
    a3.metric("Fraudulent Transactions", f"{stats['fraudulent_transactions']:,}")
    a4.metric("Fraud %", f"{stats['fraud_percentage']}%")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(eda.class_distribution_chart(df), use_container_width=True)
    with c2:
        st.plotly_chart(eda.time_distribution_chart(df), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(eda.amount_distribution_chart(df), use_container_width=True)
    with c4:
        st.plotly_chart(eda.fraud_amount_distribution_chart(df), use_container_width=True)

    with st.spinner("Computing correlation heatmap..."):
        st.plotly_chart(eda.correlation_heatmap(df), use_container_width=True)

    st.plotly_chart(eda.fraud_trend_over_time_chart(df), use_container_width=True)

    if model_available():
        model, scaler, metadata = get_model_artifacts()
        importances = feature_importance(model, metadata["feature_names"])
        if importances:
            st.subheader("🔬 Feature Importance")
            top_n = 15
            top_features = importances[:top_n]
            fig = px.bar(
                x=[imp for _, imp in top_features],
                y=[name for name, _ in top_features],
                orientation="h",
                labels={"x": "Importance", "y": "Feature"},
                title=f"Top {top_n} Most Influential Features",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("The current best model does not expose feature importances.")

# ===========================================================================
# PAGE 4 - MODEL PERFORMANCE
# ===========================================================================
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    if not model_available():
        st.warning(
            "⚠️ No trained model found. Please run `python src/train_model.py` first."
        )
        st.stop()

    model, scaler, metadata = get_model_artifacts()
    all_models = get_all_models()

    from src.data_preprocessing import full_preprocessing_pipeline
    from src.evaluate_model import evaluate_classifier, select_best_model

    with st.spinner("Evaluating model(s) on a held-out test split..."):
        pipeline_output = full_preprocessing_pipeline(DATA_PATH)
        X_test, y_test = pipeline_output["X_test"], pipeline_output["y_test"]

        results = {}
        if all_models:
            for name, m in all_models.items():
                results[name] = evaluate_classifier(m, X_test, y_test)
        else:
            results[metadata.get("best_model_name", "Model")] = evaluate_classifier(
                model, X_test, y_test
            )

    st.subheader("📋 Model Comparison")
    comparison_rows = []
    for name, metrics in results.items():
        comparison_rows.append(
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    best_name = select_best_model(results) if len(results) > 1 else list(results.keys())[0]
    st.success(f"🏆 **Best Model:** {best_name}")
    st.caption(
        "Selected by ranking fraud-class F1-score first, with ROC-AUC as a "
        "tiebreaker — not by raw accuracy, since accuracy is misleading on "
        "an imbalanced dataset like this one."
    )

    st.markdown("---")
    st.subheader(f"Detailed Metrics — {best_name}")
    best_metrics = results[best_name]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
    m2.metric("Precision", f"{best_metrics['precision']:.4f}")
    m3.metric("Recall", f"{best_metrics['recall']:.4f}")
    m4.metric("F1 Score", f"{best_metrics['f1_score']:.4f}")
    m5.metric("ROC-AUC", f"{best_metrics['roc_auc']:.4f}")

    # Confusion matrix
    st.subheader("🔢 Confusion Matrix")
    cm = np.array(best_metrics["confusion_matrix"])
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted Genuine", "Predicted Fraud"],
        y=["Actual Genuine", "Actual Fraud"],
        color_continuous_scale="Blues",
        title=f"Confusion Matrix — {best_name}",
    )
    st.plotly_chart(cm_fig, use_container_width=True)
    with st.expander("ℹ️ How to read this"):
        st.write(
            "- **TN** (top-left): genuine transactions correctly identified as genuine.\n"
            "- **FP** (top-right): genuine transactions incorrectly flagged as fraud.\n"
            "- **FN** (bottom-left): fraud transactions incorrectly missed.\n"
            "- **TP** (bottom-right): fraud transactions correctly caught."
        )

    # ROC curve
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = get_roc_curve_data(y_test, best_metrics["y_proba"])
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{best_name} (AUC={best_metrics['roc_auc']:.4f})"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash")))
    roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(roc_fig, use_container_width=True)

    # Precision-Recall curve
    st.subheader("📉 Precision-Recall Curve")
    precision, recall, _ = get_precision_recall_curve_data(y_test, best_metrics["y_proba"])
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=best_name))
    pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(pr_fig, use_container_width=True)

    st.caption(
        "ℹ️ On highly imbalanced data, the Precision-Recall curve is often more "
        "informative than the ROC curve for judging fraud-detection quality."
    )
