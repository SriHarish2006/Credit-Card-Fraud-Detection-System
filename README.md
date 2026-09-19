# 💳 Credit Card Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Status">
</p>

<p align="center">
  <b>🚨 An intelligent machine-learning system for detecting fraudulent credit-card transactions.</b>
</p>

---

## 📌 Overview

The **Credit Card Fraud Detection System** is a Machine Learning-based application designed to classify credit-card transactions as either **Genuine** or **Fraudulent**.

The project focuses on one of the major challenges in financial fraud detection: **highly imbalanced datasets**, where fraudulent transactions represent only a small percentage of total transactions.

To address this challenge, the system uses techniques such as **SMOTE** and **class-weighted learning**, along with classification algorithms including **Logistic Regression** and **Random Forest**.

A user-friendly **Streamlit dashboard** provides transaction predictions, fraud probabilities, analytics, and model performance metrics.

---

## 🎯 Objectives

- 🔍 Detect fraudulent credit-card transactions
- ⚖️ Handle highly imbalanced transaction data
- 🤖 Train multiple Machine Learning classification models
- 📊 Compare model performance using appropriate evaluation metrics
- 🚨 Predict whether a transaction is Genuine or Fraudulent
- 📈 Display fraud probability
- 📊 Provide interactive fraud analytics
- 🖥️ Build an easy-to-use Streamlit web application

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 Fraud Detection | Classifies transactions as Genuine or Fraudulent |
| ⚖️ Imbalance Handling | Uses SMOTE and class-weighted learning |
| 🤖 Multiple Models | Logistic Regression and Random Forest |
| 📊 Model Evaluation | Accuracy, Precision, Recall, F1-Score & ROC-AUC |
| 🚨 Probability Prediction | Displays predicted fraud probability |
| 📈 Interactive Analytics | Visualize transaction and fraud statistics |
| 🔥 Feature Importance | Identify influential transaction features |
| 🧩 Confusion Matrix | Analyze classification errors |
| 📱 Streamlit UI | Interactive web-based interface |
| 💾 Model Persistence | Saves trained models using Joblib |

---

# 🧠 Machine Learning Workflow

```text
                 ┌─────────────────────┐
                 │  Credit Card Data   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Exploratory Data    │
                 │      Analysis       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Train/Test Split    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Class Imbalance     │
                 │ Handling            │
                 │ SMOTE / Class Weight│
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        ┌─────────────────┐   ┌─────────────────┐
        │    Logistic     │   │  Random Forest  │
        │   Regression    │   │   Classifier    │
        └────────┬────────┘   └────────┬────────┘
                 │                     │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Model Evaluation    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Select Best Model   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Save Model + Scaler │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit Web App   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Fraud / Genuine     │
                 │ Prediction          │
                 └─────────────────────┘


