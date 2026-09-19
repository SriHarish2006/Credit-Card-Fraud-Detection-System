# 💳 Credit Card Fraud Detection System

### Machine Learning-Based Fraud Detection with Imbalanced Data Handling

<p align="center">
  <b>Detect • Analyze • Predict • Protect</b>
</p>

<p align="center">
  An end-to-end machine learning project for identifying fraudulent credit card transactions using classification algorithms and techniques for handling highly imbalanced datasets.
</p>

---

## 🚀 Project Overview

The **Credit Card Fraud Detection System** is a machine learning-based application designed to classify credit card transactions as **Genuine** or **Fraudulent**.

Credit card fraud datasets are highly imbalanced because fraudulent transactions usually represent only a small fraction of all transactions. This project addresses that challenge through appropriate **data preprocessing and imbalance-handling techniques**, followed by machine learning classification.

The system provides a user-friendly interface through **Streamlit**, allowing users to provide transaction information and obtain a fraud prediction.

---

## 🎯 Key Objectives

* 🔍 Detect potentially fraudulent credit card transactions
* 🤖 Apply machine learning for binary classification
* ⚖️ Handle highly imbalanced transaction data
* 🧹 Perform data preprocessing and feature preparation
* 🌲 Train classification models such as Random Forest
* 📈 Evaluate model performance using appropriate classification metrics
* 🖥️ Provide an interactive Streamlit interface
* 💾 Save and reuse trained machine learning models
* 📊 Present prediction results in an understandable format

---

## ✨ Features

| Feature                 | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| 🔐 Fraud Detection      | Classifies transactions as Genuine or Fraudulent        |
| 🤖 Machine Learning     | Uses supervised classification algorithms               |
| ⚖️ Imbalance Handling   | Supports techniques such as SMOTE and class weighting   |
| 📊 Model Evaluation     | Precision, Recall, F1-Score, Accuracy and other metrics |
| 🌐 Streamlit UI         | Interactive web-based prediction interface              |
| 💾 Saved Models         | Trained models can be reused without retraining         |
| 📈 Visualization        | Helps analyze model and transaction behavior            |
| 🧩 Modular Architecture | Separates frontend, backend and ML components           |

---

# 🏗️ System Architecture

```text
                   ┌─────────────────────┐
                   │      User Input     │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Streamlit UI      │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Data Preprocessing  │
                   │ & Feature Handling  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Trained ML Model    │
                   │ Random Forest / LR  │
                   └──────────┬──────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Prediction Result    │
                 │                          │
                 │  🟢 Genuine Transaction  │
                 │          OR              │
                 │  🔴 Fraudulent           │
                 └──────────────────────────┘
```

---

# 🔄 Machine Learning Workflow

```text
Dataset
   │
   ▼
Data Exploration
   │
   ▼
Data Cleaning
   │
   ▼
Feature Preparation
   │
   ▼
Train / Test Split
   │
   ▼
Imbalanced Data Handling
   │
   ├── SMOTE
   │
   └── Class Weights
   │
   ▼
Model Training
   │
   ├── Logistic Regression
   │
   └── Random Forest
   │
   ▼
Model Evaluation
   │
   ▼
Model Serialization
   │
   ▼
Streamlit Application
   │
   ▼
Fraud Prediction
```

---

# 🧠 Machine Learning Models

## 1. Logistic Regression

Logistic Regression is used as a classification model for predicting whether a transaction belongs to the genuine or fraudulent class.

### Advantages

* Simple and interpretable
* Efficient for binary classification
* Provides probability-based predictions
* Useful as a baseline model

---

## 2. Random Forest 🌲

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.

### Advantages

* Handles nonlinear relationships
* Robust against overfitting compared with individual decision trees
* Can provide feature importance
* Suitable for complex classification patterns

---

# ⚖️ Handling Class Imbalance

Fraud detection datasets typically contain significantly more genuine transactions than fraudulent transactions.

For example:

```text
Genuine Transactions      ████████████████████████████████████████
Fraudulent Transactions   █
```

If imbalance is ignored, a model may achieve high overall accuracy while performing poorly on fraud cases.

This project therefore considers techniques such as:

### SMOTE

**Synthetic Minority Over-sampling Technique (SMOTE)** creates synthetic examples of the minority class to improve its representation during training.

### Class Weighting

Class weights can assign greater importance to the minority fraud class during model training.

---

# 📊 Model Evaluation

Fraud detection should not rely only on accuracy.

The project evaluates classification performance using metrics such as:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**
* **ROC-AUC**, when implemented

### Evaluation Table

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |        — |         — |      — |        — |       — |
| Random Forest       |        — |         — |      — |        — |       — |

> Replace the `—` values with your actual experimental results before final submission.

---

# 🎨 Streamlit Application

The project includes an interactive Streamlit interface for making fraud predictions.

### Application Flow

```text
Transaction Details
        │
        ▼
   Input Features
        │
        ▼
  Preprocessing
        │
        ▼
   ML Prediction
        │
        ▼
 ┌───────────────────┐
 │ Genuine / Fraud   │
 └───────────────────┘
```

### 📸 Screenshots

Add your actual project screenshots here:

#### 🏠 Application Interface

> 📷 `screenshots/home.png`

#### 💳 Transaction Prediction

> 📷 `screenshots/prediction.png`

#### 📊 Results / Dashboard

> 📷 `screenshots/results.png`

---

# 🛠️ Technology Stack

| Category            | Technologies                |
| ------------------- | --------------------------- |
| Language            | 🐍 Python                   |
| Machine Learning    | Scikit-learn                |
| Imbalanced Learning | imbalanced-learn / SMOTE    |
| Data Processing     | Pandas, NumPy               |
| Visualization       | Matplotlib, Seaborn, Plotly |
| Web Interface       | Streamlit                   |
| Model Storage       | Joblib / Pickle             |
| Development         | VS Code / Jupyter Notebook  |
| Version Control     | Git & GitHub                |

---

# 📁 Project Structure

```text
fraud-detection-system/
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── ...
│
├── ml/
│   ├── dataset/
│   │   └── ...
│   │
│   ├── saved_models/
│   │   └── ...
│   │
│   └── ...
│
├── streamlit_app/
│   ├── app.py
│   ├── requirements.txt
│   └── ...
│
├── screenshots/
│   ├── home.png
│   ├── prediction.png
│   └── results.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Update the structure if your final repository contains different files or folders.

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/SriHarish2006/fraud-detection-system.git
```

```bash
cd fraud-detection-system
```

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
python -m pip install -r requirements.txt
```

If the Streamlit application has its own requirements file:

```bash
python -m pip install -r streamlit_app/requirements.txt
```

---

# ▶️ Running the Streamlit Application

From the project root:

```bash
streamlit run streamlit_app/app.py
```

Then open the local Streamlit URL displayed in the terminal.

Typically:

```text
http://localhost:8501
```

---

# 🔬 Example Prediction

### Input

```text
Transaction Features
        ↓
Feature Preprocessing
        ↓
Trained Random Forest Model
```

### Output

```text
┌──────────────────────────────┐
│     🔴 FRAUDULENT            │
│                              │
│ Potential fraudulent         │
│ transaction detected.        │
└──────────────────────────────┘
```

or

```text
┌──────────────────────────────┐
│     🟢 GENUINE               │
│                              │
│ Transaction classified as    │
│ genuine.                     │
└──────────────────────────────┘
```

---

# 📌 Important Considerations

A machine learning prediction is not a guarantee that a transaction is actually fraudulent.

Real-world financial fraud detection systems require additional components such as:

* Real-time transaction monitoring
* Secure financial infrastructure
* Authentication and authorization
* Data privacy controls
* Fraud investigation workflows
* Continuous model monitoring
* Model retraining
* Production-grade APIs

This project is intended as an **academic and machine learning demonstration**.

---

# 🌟 Advantages

* Automated transaction classification
* Machine learning-based detection
* Addresses class imbalance
* Interactive prediction interface
* Modular project structure
* Reusable trained models
* Suitable for experimentation and academic learning
* Can be extended toward real-time fraud monitoring

---

# 🔮 Future Enhancements

Future versions can include:

* ⚡ Real-time transaction monitoring
* 🧠 XGBoost / LightGBM models
* 🤖 Deep learning approaches
* 🔎 Explainable AI using SHAP
* 🔐 User authentication
* 🗄️ Database integration
* 📧 Fraud alert notifications
* 📱 Mobile-friendly interface
* ☁️ Cloud deployment
* 🔄 Automated model retraining
* 📊 Advanced fraud analytics dashboard
* 🔌 Production-ready REST API

---

# 💼 Real-World Applications

The concept can be applied to:

* 🏦 Banking systems
* 💳 Credit card companies
* 💰 FinTech platforms
* 🛒 E-commerce payment systems
* 📱 Digital wallets
* 💻 Online payment gateways
* 🏢 Financial institutions

---

# 🎓 Project Information

| Detail           | Information                        |
| ---------------- | ---------------------------------- |
| Project          | Credit Card Fraud Detection System |
| Domain           | Machine Learning / Data Science    |
| Problem Type     | Binary Classification              |
| Main Challenge   | Highly Imbalanced Data             |
| Primary Models   | Logistic Regression, Random Forest |
| Interface        | Streamlit                          |
| Language         | Python                             |
| Project Category | Academic / Internship Project      |

---

# 👨‍💻 Author

### Sri Harish

**B.E. Computer Science and Engineering**

   Ph.No:+91-9489231147

🔗 LinkedIn: **[https://www.linkedin.com/in/sri-harish-2b34a930a/]**

---

# 📜 License

This project is intended for **educational and academic purposes**.

Add an appropriate open-source license if you intend to distribute the project under one.

---

<p align="center">

### 💳 Detect Fraud. Analyze Transactions. Build Safer Payment Systems.

⭐ **If you find this project useful, consider giving the repository a star!**

</p>
