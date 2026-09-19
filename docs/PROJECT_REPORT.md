# Project Report: Credit Card Fraud Detection System Using Machine Learning

---

## Chapter 1 — Introduction

Credit and debit cards are among the most widely used payment methods
worldwide, which also makes them a major target for fraud. Manual review
of every transaction is impossible at scale, so financial institutions
increasingly rely on automated, data-driven systems to flag suspicious
activity in real time. This project builds a Machine Learning pipeline
that classifies credit-card transactions as genuine or fraudulent, and
presents the results through an interactive web application, as a
practical demonstration of applied classification, imbalanced-data
handling, and model evaluation techniques learned in coursework.

## Chapter 2 — Literature / Background

Fraud detection is a well-studied application of supervised Machine
Learning. Classical approaches range from rule-based systems (manually
written "if-this-then-flag" heuristics) to statistical and ML models such
as Logistic Regression, Decision Trees, Random Forests, Gradient
Boosting, and, more recently, deep neural networks and graph-based
methods. A recurring challenge across the literature is **severe class
imbalance** — fraudulent transactions are always a tiny minority — which
has motivated techniques like cost-sensitive learning (class weighting)
and resampling methods such as SMOTE. Evaluation in this domain
consistently emphasizes precision, recall, F1-score, and ROC-AUC over
raw accuracy, since accuracy is trivially high and uninformative on
imbalanced data.

## Chapter 3 — Problem Statement and Objectives

**Problem statement:** Given an anonymised set of transaction features,
build a model that reliably distinguishes fraudulent transactions from
genuine ones, despite fraud accounting for a very small fraction of all
transactions.

**Objectives:**
1. Build a robust, reproducible data-preprocessing pipeline.
2. Correctly handle class imbalance without introducing data leakage.
3. Train and fairly compare at least two classification models.
4. Evaluate models using metrics appropriate for imbalanced classification.
5. Deliver results through an interactive, explainable web application.

## Chapter 4 — System Analysis

**Input:** A CSV file of transactions with columns `Time`, `V1`–`V28`
(PCA-anonymised numeric features), `Amount`, and target `Class`.

**Output:** For each transaction, a predicted class (Genuine/Fraudulent)
and associated class probabilities, plus dataset-wide analytics and model
performance reporting.

**Constraints:** The dataset is highly imbalanced (~0.17% fraud); no
personally identifiable or raw card information is present or required;
the system must run on standard consumer hardware without specialized
infrastructure.

**Feasibility:** The chosen stack (Pandas, Scikit-learn,
imbalanced-learn, Streamlit) is free, open-source, well-documented, and
appropriately scoped for the dataset size and project timeline.

## Chapter 5 — System Design

The system follows a linear pipeline architecture, cleanly separated into
independent modules:

- `data_preprocessing.py` — loading, validation, cleaning, splitting, scaling.
- `eda.py` — statistics and chart-generation functions, reused by both the
  notebook and the web app.
- `train_model.py` — orchestrates SMOTE, model training, evaluation, and
  persistence; run once from the command line.
- `evaluate_model.py` — metric computation and model-selection logic,
  shared between training and the app.
- `prediction.py` — loads saved artifacts and runs inference on new input.
- `streamlit_app.py` — the user-facing presentation layer, calling into
  the modules above.

This separation means the ML logic is testable independently of the UI,
and the UI never re-implements preprocessing logic — it always calls the
same functions used during training, preventing train/serve mismatches.

## Chapter 6 — Methodology

1. **Data validation** — verify required columns exist and the target is
   binary before any processing begins.
2. **Data cleaning** — impute missing values, remove duplicate rows.
3. **Stratified train/test split** — preserves the true fraud ratio in
   both splits despite the severe imbalance.
4. **Feature scaling** — `StandardScaler` fit on training data only.
5. **Imbalance handling** — SMOTE applied strictly to the training split,
   combined with `class_weight="balanced"` in both models.
6. **Model training** — Logistic Regression (linear baseline) and Random
   Forest (ensemble, non-linear).
7. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC,
   Confusion Matrix, ROC Curve, Precision-Recall Curve, all computed on
   the untouched, realistically-imbalanced test set.
8. **Model selection** — automatic, ranked by fraud-class F1-score with
   ROC-AUC as a tiebreaker.
9. **Persistence & serving** — best model, scaler, and feature metadata
   saved via Joblib and loaded (not retrained) by the Streamlit app.

## Chapter 7 — Implementation

The project is implemented in Python 3, using Pandas/NumPy for data
handling, Scikit-learn and imbalanced-learn for modeling, Plotly for
interactive charts, and Streamlit for the web interface. Code is
organized into small, documented functions rather than large monolithic
scripts, with docstrings explaining the purpose of each preprocessing and
evaluation step — particularly around the imbalance-handling logic, which
is the most conceptually important part of the project. Error handling is
applied throughout (missing dataset, missing model, invalid prediction
input) so the application fails gracefully with clear messages rather
than raw stack traces.

## Chapter 8 — Machine Learning Models

**Logistic Regression** models the log-odds of the fraud class as a
linear combination of input features. It is fast to train, easy to
interpret via its coefficients, and serves as a sanity-check baseline.

**Random Forest** is an ensemble of many decision trees, each trained on
a bootstrapped sample of the data with a random subset of features
considered at each split; predictions are aggregated across trees. It
naturally captures non-linear relationships and feature interactions, and
provides a built-in feature-importance ranking, which is used in the
Analytics page of the app.

Both models are configured with `class_weight="balanced"` as an
additional safeguard alongside SMOTE.

## Chapter 9 — Results and Evaluation

All results are computed dynamically at training time from whichever
dataset is placed in `data/creditcard.csv` and written to
`outputs/reports/training_report.json` — no numbers are hard-coded
anywhere in the codebase. In development runs on the standard 284,807-row
dataset (492 fraud cases, ≈0.173%), Random Forest consistently
outperformed Logistic Regression on fraud-class F1-score and ROC-AUC and
was automatically selected as the best model, while Logistic Regression
achieved noticeably higher recall at the cost of substantially lower
precision. This illustrates the classic precision/recall trade-off:
Logistic Regression flags many more transactions as suspicious (catching
more real fraud but generating more false alarms), while Random Forest
strikes a better overall balance. Exact figures for a given run are
available in the comparison table on the app's Model Performance page and
in `training_report.json`.

## Chapter 10 — Screenshots / User Interface

*(Insert screenshots here after running the application locally: the
Dashboard, the Fraud Prediction form and result, the Analytics charts,
and the Model Performance / confusion-matrix view.)*

## Chapter 11 — Advantages and Limitations

**Advantages:**
- Clean, modular, reproducible pipeline with no data leakage.
- Fraud-appropriate evaluation (not just accuracy).
- Transparent, automatic model selection.
- Interactive, explainable web interface suitable for demonstration.

**Limitations:**
- Trained on one historical dataset; may drift from current fraud patterns.
- PCA-anonymised features limit business-level interpretability.
- Manual entry of `V1`–`V28` in the demo UI is unrealistic for production use.
- Not designed for high-throughput, low-latency production deployment.

## Chapter 12 — Future Enhancements

Real-time transaction monitoring, a REST API for programmatic predictions,
database integration for transaction history, automated alerting,
additional ensemble models (XGBoost/LightGBM) for comparison, decision
threshold optimization, SHAP-based explainability, model monitoring for
drift, cloud deployment, and authentication/role-based access control.

## Chapter 13 — Conclusion

This project demonstrates a complete, honest Machine Learning workflow
for a genuinely hard, imbalanced classification problem: careful data
validation and cleaning, leak-free preprocessing, principled
class-imbalance handling via SMOTE and class weighting, multi-model
comparison using fraud-appropriate metrics, transparent automatic model
selection, and a usable interactive front-end for exploration and live
predictions — all scoped appropriately for a college minor project while
following practices that mirror real-world fraud-detection systems.
