# 💳 Credit Card Fraud Detection System Using Machine Learning

A complete, end-to-end Machine Learning project that detects fraudulent
credit-card transactions and exposes the results through an interactive
Streamlit web application.

> ⚠️ **Educational project only.** This is a college minor project, not a
> production banking fraud-prevention system. Never enter real credit-card
> numbers or real customer data anywhere in this application.

---

## Overview

Credit-card fraud costs billions of dollars annually, and fraudulent
transactions make up a tiny fraction of all transactions — which makes
detecting them a genuinely hard *imbalanced classification* problem. This
project builds, evaluates, and compares two Machine Learning models
(Logistic Regression and Random Forest) to classify transactions as
**Genuine** or **Fraudulent**, and wraps the result in a clean, explorable
Streamlit dashboard.

## Problem Statement

Given anonymised transaction features (`Time`, `V1`–`V28`, `Amount`),
predict whether a transaction is fraudulent (`Class = 1`) or genuine
(`Class = 0`), despite fraud representing well under 1% of all transactions
in the dataset.

## Objectives

- Build a clean, reproducible ML pipeline from raw CSV to a saved model.
- Correctly handle severe class imbalance without leaking information
  between train and test data.
- Compare multiple models using metrics that actually matter for fraud
  detection (not just accuracy).
- Present results through an interactive, explainable web application.

## Features

- Automatic dataset validation, cleaning, and quality reporting.
- Exploratory Data Analysis with interactive Plotly charts.
- Class-imbalance handling via **SMOTE** (train-only) and `class_weight="balanced"`.
- Two trained models — Logistic Regression and Random Forest — with a
  transparent, metric-driven model-selection rule.
- Full evaluation suite: Accuracy, Precision, Recall, F1-score, ROC-AUC,
  Confusion Matrix, ROC Curve, Precision-Recall Curve.
- Feature importance visualization (Random Forest).
- A 4-page Streamlit app: Dashboard, Fraud Prediction, Analytics, Model
  Performance.
- Graceful error handling throughout (missing files, bad input, etc.).

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, imbalanced-learn (SMOTE) |
| Models | Logistic Regression, Random Forest |
| Visualization | Plotly, Matplotlib |
| Web app | Streamlit |
| Model persistence | Joblib |
| Development | Jupyter Notebook, VS Code |
| Version control | Git, GitHub |

## Dataset

The project expects the well-known **Credit Card Fraud Detection** dataset
(anonymised European cardholder transactions, September 2013), placed at:

```
data/creditcard.csv
```

Expected columns: `Time`, `V1`...`V28` (PCA-anonymised features), `Amount`,
and the target column `Class` (`0` = genuine, `1` = fraud).

The dataset used during development contains **284,807 transactions**, of
which **492 (≈0.173%)** are fraudulent — a strongly imbalanced dataset,
which is the central challenge this project addresses. All statistics
shown in the app are computed dynamically from whatever CSV is placed in
`data/` — nothing is hard-coded.

## System Architecture

```
Dataset (CSV)
      │
      ▼
Data Validation & Cleaning  (src/data_preprocessing.py)
      │
      ▼
Exploratory Data Analysis   (src/eda.py, notebooks/)
      │
      ▼
Train/Test Split (stratified) + Feature Scaling
      │
      ▼
SMOTE (training data only)  (src/train_model.py)
      │
      ▼
Model Training: Logistic Regression + Random Forest
      │
      ▼
Evaluation & Model Comparison (src/evaluate_model.py)
      │
      ▼
Best Model Selection → Save (Joblib)  (models/)
      │
      ▼
Streamlit Web Application  (app/streamlit_app.py)
```

## Machine Learning Workflow

1. **Load & validate** the dataset — check shape, columns, dtypes, missing
   values, duplicates, and that `Class` only contains 0/1.
2. **Clean** the dataset — impute any missing numeric values with the
   median, drop duplicate rows.
3. **Split** features/target, then **stratified train/test split** so the
   rare fraud class is proportionally represented in both sets.
4. **Scale** features with `StandardScaler`, fit on the training set only.
5. **Balance** the training set with **SMOTE** (test set is left untouched,
   at its real-world imbalanced distribution).
6. **Train** Logistic Regression (`class_weight="balanced"`) and Random
   Forest (`class_weight="balanced"`, 200 trees).
7. **Evaluate** both models on the untouched, imbalanced test set.
8. **Select** the best model by fraud-class F1-score (ROC-AUC as
   tiebreaker) — not accuracy.
9. **Save** the winning model + scaler + feature order with Joblib.
10. **Serve** predictions through Streamlit, loading the saved model
    instead of retraining on every page load.

## Data Preprocessing

See `src/data_preprocessing.py`. Key points:

- Missing target rows are dropped; missing feature values are median-imputed.
- Duplicate rows are removed (the raw dataset commonly contains ~1,000+
  exact duplicate rows).
- `StandardScaler` is fit **only** on the training split and then applied
  to both train and test, to avoid leaking test-set statistics into
  training.

## Imbalanced Data Handling

Fraud is rare — in this dataset, roughly 1 in 578 transactions. A naive
model can score >99% accuracy by predicting "genuine" every time, while
being completely useless at catching fraud. Two complementary techniques
are used:

1. **`class_weight="balanced"`** on both models, so misclassifying a fraud
   case is penalized more heavily during training.
2. **SMOTE (Synthetic Minority Over-sampling Technique)**, applied to the
   training data only, which synthesizes new, plausible fraud examples by
   interpolating between existing minority-class neighbors.

### SMOTE Explanation

SMOTE does **not** duplicate existing fraud rows. For each minority-class
(fraud) sample, it finds its nearest minority-class neighbors in feature
space and creates new synthetic samples along the line segments connecting
them. This gives the model more varied fraud examples to learn from,
without simply copy-pasting the same handful of cases.

**Critically, SMOTE is applied only to the training data, after the
train/test split** — never before. Applying it before splitting would let
synthetic near-duplicates of the same original fraud case end up in both
the training and test sets, artificially inflating test performance (a
form of **data leakage**). The test set always keeps the real, original
class distribution so evaluation reflects real-world conditions.

## Machine Learning Models

**Logistic Regression** — a simple, fast, interpretable linear baseline.
Good at showing whether a linear decision boundary can separate the
classes at all.

**Random Forest Classifier** — an ensemble of decision trees that can
capture non-linear relationships between the anonymised `V1`–`V28`
features, generally the stronger performer on this kind of tabular data,
and provides feature-importance scores "for free."

## Evaluation Metrics

Accuracy alone is misleading here — with fraud at ~0.17% of transactions,
a model predicting "genuine" for everything scores ~99.8% accuracy while
catching **zero** fraud. This project reports and prioritizes:

- **Precision** — of the transactions flagged as fraud, how many actually were?
- **Recall** — of the actual fraud cases, how many did the model catch?
- **F1-score** — harmonic mean of precision and recall.
- **ROC-AUC** — overall ability to rank fraud above genuine transactions.
- **Confusion Matrix**, **ROC Curve**, **Precision-Recall Curve** for a
  fuller picture.

The best model is selected primarily by **fraud-class F1-score**, with
ROC-AUC as a tiebreaker — never by raw accuracy.

## Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_detection_model.pkl   # best model (Joblib)
│   ├── scaler.pkl                  # fitted StandardScaler
│   ├── metadata.pkl                # feature order + best model name
│   └── all_models.pkl              # every trained model, for comparison
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── prediction.py
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
│
├── outputs/
│   ├── figures/
│   └── reports/
│       └── training_report.json    # generated after training
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation

```bash
git clone <repository-url>
cd credit-card-fraud-detection

python -m venv venv
```

Activate the virtual environment:

- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

Place `creditcard.csv` inside the `data/` folder (see **Dataset** above).

## How to Run

**1. Train the model** (run once, or whenever the dataset changes):

```bash
python src/train_model.py
```

This prints dataset statistics, the SMOTE before/after class distribution,
model training progress, evaluation metrics for both models, and saves the
winning model to `models/`.

**2. Launch the Streamlit application:**

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Streamlit Application

- **🏠 Dashboard** — headline metrics (total/genuine/fraud transactions,
  fraud rate, model recall/F1/ROC-AUC) plus summary charts.
- **🔍 Fraud Prediction** — enter (or auto-fill a random real sample of)
  transaction details and get a live prediction with class probabilities.
- **📊 Analytics** — deeper EDA: amount/time distributions, correlation
  heatmap, fraud-rate trend over time, and feature importance.
- **🤖 Model Performance** — full model comparison table, confusion
  matrix, ROC curve, and Precision-Recall curve for the best model.

## Screenshots

*(Add screenshots here after running the app locally, e.g.
`outputs/figures/dashboard.png`, `outputs/figures/prediction.png`.)*

## Results

Results are **not hard-coded** — every number in the app and in
`outputs/reports/training_report.json` is computed live from whatever
dataset is placed in `data/creditcard.csv`. Re-run `python
src/train_model.py` to regenerate them. On the standard 284,807-row
dataset, Random Forest with SMOTE-balanced training data was selected as
the best model based on fraud-class F1-score.

## Future Enhancements

- Real-time transaction monitoring / streaming pipeline
- REST API for programmatic predictions
- Database integration for transaction history
- Alert/notification system for flagged transactions
- Additional ensemble models (XGBoost, LightGBM) for comparison
- Decision-threshold optimization (beyond the default 0.5 cutoff)
- Explainable AI: SHAP-based per-prediction explanations
- Model monitoring / drift detection in production
- Cloud deployment
- Authentication and role-based access control

## Limitations

- Trained on a single historical (2013) dataset; may not generalize to
  current fraud patterns without retraining on fresh data.
- `V1`–`V28` are PCA-anonymised, so individual feature meaning isn't
  interpretable in a business sense.
- The prediction UI requires manually entering already-anonymised PCA
  features, which is unrealistic for a real deployment (a production
  system would compute these automatically upstream).
- Not tuned for extreme low-latency, high-throughput production use.
- This is not a substitute for a certified banking fraud-detection system.

## Conclusion

This project demonstrates a complete, realistic ML workflow for a hard,
imbalanced classification problem — from raw data validation through
leak-free preprocessing, principled class-imbalance handling, multi-model
comparison with fraud-appropriate metrics, and a usable interactive
front-end — while staying scoped appropriately for a college minor
project.

## Author

**[Your Name Here]**
[Your Institution / Course Here]
[Your Email Here]
