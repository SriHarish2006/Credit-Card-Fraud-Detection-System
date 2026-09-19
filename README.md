# 💳 Credit Card Fraud Detection System

### 🔐 Machine Learning-Based Fraud Detection with Imbalanced Data Handling

> An end-to-end machine learning system designed to identify **fraudulent credit card transactions** from legitimate transactions using classification algorithms and techniques for handling highly imbalanced datasets.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Status">
</p>

---

## 📌 Overview

Credit card fraud is a major challenge in digital payment systems because fraudulent transactions typically represent only a **small fraction of all transactions**.

This project uses **Machine Learning classification techniques** to distinguish between:

* 🟢 **Genuine Transactions**
* 🔴 **Fraudulent Transactions**

The system focuses particularly on the **class imbalance problem**, where legitimate transactions significantly outnumber fraudulent transactions.

The project includes data preprocessing, imbalance handling, model training, evaluation, and a user-friendly prediction interface.

---

## 🎯 Objectives

* Detect fraudulent credit card transactions automatically.
* Classify transactions as **Genuine** or **Fraudulent**.
* Handle highly imbalanced transaction data.
* Apply suitable data preprocessing techniques.
* Train and compare machine learning classification models.
* Evaluate models using meaningful classification metrics.
* Provide an easy-to-use interface for fraud prediction.
* Create a foundation that can be extended toward real-time fraud monitoring.

---

## 🧠 Machine Learning Approach

The project follows an end-to-end ML pipeline:

```text
📊 Transaction Dataset
        ↓
🧹 Data Preprocessing
        ↓
⚖️ Imbalanced Data Handling
        ↓
✂️ Train / Test Split
        ↓
🤖 Model Training
        ↓
📈 Model Evaluation
        ↓
💾 Model Serialization
        ↓
🌐 Streamlit Application
        ↓
🔍 Fraud / Genuine Prediction
```

---

## ⚖️ Handling Class Imbalance

One of the main challenges in credit card fraud detection is **class imbalance**.

A typical fraud dataset contains a very large number of genuine transactions and a much smaller number of fraudulent transactions.

Simply optimizing for accuracy can therefore produce misleading results.

This project explores techniques such as:

### 🔹 SMOTE

**Synthetic Minority Over-sampling Technique (SMOTE)** generates synthetic samples for the minority class to improve the model's ability to learn fraudulent transaction patterns.

### 🔹 Class Weights

Class-weighting techniques can assign greater importance to the minority fraud class during model training.

```text
Highly Imbalanced Data
          ↓
   SMOTE / Class Weights
          ↓
Improved Minority-Class Learning
          ↓
Fraud Detection Model
```

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression

Logistic Regression is used as a classification approach for predicting whether a transaction belongs to the genuine or fraudulent class.

**Key characteristics:**

* Binary classification
* Probability-based prediction
* Simple and interpretable
* Useful as a baseline model

### 2️⃣ Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.

**Key characteristics:**

* Handles nonlinear relationships
* Robust classification performance
* Supports feature importance analysis
* Suitable for complex transaction patterns

---

## 📊 Model Evaluation

Because fraud detection is an imbalanced classification problem, multiple evaluation metrics are considered.

| Metric              | Purpose                                           |
| ------------------- | ------------------------------------------------- |
| 🎯 Accuracy         | Overall percentage of correct predictions         |
| 🔎 Precision        | How many predicted fraud cases are actually fraud |
| 🚨 Recall           | How many actual fraud cases are detected          |
| ⚖️ F1-Score         | Balance between precision and recall              |
| 📈 ROC-AUC          | Measures classification discrimination            |
| 🧮 Confusion Matrix | Shows TP, TN, FP and FN                           |

> **Important:** Actual performance values should be taken from the final trained model rather than being hard-coded into the README.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Transaction Data  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Imbalance Handling  │
                    │   SMOTE / Weights   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   ML Model Training │
                    │ Logistic Regression │
                    │    Random Forest    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │  Model Evaluation   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   Saved ML Model    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Streamlit Interface │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Fraud / Genuine     │
                    │     Prediction      │
                    └─────────────────────┘
```

---

## 🧩 Project Modules

| Module                | Description                            |
| --------------------- | -------------------------------------- |
| 📥 Data Collection    | Loads the transaction dataset          |
| 🧹 Data Preprocessing | Cleans and prepares transaction data   |
| ⚖️ Imbalance Handling | Applies SMOTE/class-weight techniques  |
| 🤖 Model Training     | Trains classification algorithms       |
| 📊 Model Evaluation   | Calculates classification metrics      |
| 💾 Model Saving       | Stores trained model for inference     |
| 🔍 Prediction         | Predicts fraud or genuine transactions |
| 🖥️ Streamlit UI      | Provides an interactive interface      |

---

## 🛠️ Technology Stack

### Programming

* 🐍 Python

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest

### Data Processing

* Pandas
* NumPy

### Imbalanced Data

* Imbalanced-learn
* SMOTE
* Class Weighting

### Visualization

* Matplotlib
* Seaborn
* Plotly *(if enabled in the current version)*

### Application

* Streamlit
* FastAPI *(for the backend version)*

### Development

* Git
* GitHub
* VS Code
* Jupyter Notebook

---

## 📁 Project Structure

```text
credit-card-fraud-detection-system/
│
├── 📂 backend/
│   ├── app.py
│   └── ...
│
├── 📂 frontend/
│   └── ...
│
├── 📂 ml/
│   ├── saved_models/
│   │   └── ...
│   ├── train.py
│   └── ...
│
├── 📂 streamlit_app/
│   ├── app.py
│   └── ...
│
├── 📂 data/
│   └── ...
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 .gitignore
```

> The exact files may vary depending on the version of the project uploaded to the repository.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SriHarish2006/credit-card-fraud-detection-system.git
```

```bash
cd credit-card-fraud-detection-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Navigate to the Streamlit application directory if required:

```bash
cd streamlit_app
```

Run:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## 🔌 Run the FastAPI Backend

If using the backend version:

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

---

## 🖥️ Application Features

### 🏠 Interactive Dashboard

The Streamlit interface provides a simple way to interact with the fraud detection model.

### 💳 Transaction Prediction

Users can provide transaction information and request a prediction.

### 🚨 Fraud Detection

The model produces a classification indicating whether the transaction is predicted to be:

```text
🔴 FRAUDULENT
```

or

```text
🟢 GENUINE
```

### 📊 Model-Based Analysis

The system can be extended with:

* Prediction probabilities
* Confusion matrix
* Performance metrics
* Feature analysis
* Transaction visualizations

---

## 📸 Screenshots

### 🏠 Home Page

> Add your Streamlit home-page screenshot here.

```text
[ INSERT HOME PAGE SCREENSHOT ]
```

### 🔍 Fraud Prediction

> Add your fraud-prediction screenshot here.

```text
[ INSERT FRAUD PREDICTION SCREENSHOT ]
```

### 🟢 Genuine Transaction

> Add your genuine-transaction screenshot here.

```text
[ INSERT GENUINE TRANSACTION SCREENSHOT ]
```

### 📊 Model Performance

> Add your model evaluation screenshot here.

```text
[ INSERT MODEL PERFORMANCE SCREENSHOT ]
```

---

## 🔐 Security & Practical Considerations

This project is intended for **educational and demonstration purposes**.

A production banking system would additionally require:

* Secure API authentication
* Encrypted communication
* Secure database infrastructure
* Transaction monitoring
* Model drift monitoring
* Fraud investigation workflows
* Access control
* Logging and auditing
* Continuous model validation

---

## ⚠️ Limitations

* Model performance depends on the quality and representativeness of the training data.
* Fraud patterns can change over time.
* False positives and false negatives are possible.
* A model trained on historical data may not detect previously unseen fraud patterns.
* Production deployment requires additional security and infrastructure.

---

## 🔮 Future Enhancements

* ⚡ Real-time transaction monitoring
* 🧠 XGBoost / LightGBM models
* 🤖 Deep learning-based fraud detection
* 🔍 Explainable AI using SHAP
* 📡 Real-time fraud alerts
* 📱 SMS / Email notifications
* 🗄️ Database integration
* 🔐 Authentication and authorization
* ☁️ Cloud deployment
* 📊 Advanced fraud analytics dashboard
* 🔄 Continuous model retraining
* 📈 Model drift detection

---

## 💼 Real-World Applications

The concept can be applied to:

* 🏦 Banking systems
* 💳 Credit card companies
* 💰 FinTech platforms
* 🛒 E-commerce platforms
* 📱 Digital wallets
* 💻 Online payment gateways
* 🏪 Digital financial services

---

## 🎓 Project Information

**Project:** Credit Card Fraud Detection System
**Category:** Machine Learning / Data Science
**Domain:** Financial Technology (FinTech)
**Problem Type:** Binary Classification
**Key Challenge:** Imbalanced Data
**Primary Language:** Python
**Interface:** Streamlit
**Backend:** FastAPI *(where applicable)*

---

## 👨‍💻 Developer

### Sri Harish

**B.E. Computer Science and Engineering**

Interested in:

`Data Analytics` • `Machine Learning` • `Python` • `SQL` • `Power BI` • `Artificial Intelligence`

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended primarily for **educational and academic purposes**. Add an appropriate open-source license if you plan to distribute the project for reuse.

---

<p align="center">
  <b>💳 Detect Fraud. Protect Transactions. Build Smarter Financial Systems. 🔐</b>
</p>
