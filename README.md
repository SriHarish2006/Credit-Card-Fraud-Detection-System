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


#📂 Project Structure

credit-card-fraud-detection/
│
├── 📁 data/
│   └── creditcard.csv
│
├── 📁 models/
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
│
├── 📁 notebooks/
│   └── fraud_detection_analysis.ipynb
│
├── 📁 src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── prediction.py
│
├── 📁 app/
│   ├── __init__.py
│   └── streamlit_app.py
│
├── 📁 outputs/
│   ├── figures/
│   └── reports/
│
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
└── 📄 LICENSE


#🛠️ Tech Stack
Programming Language

🐍 Python

Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
imbalanced-learn
Logistic Regression
Random Forest
Data Visualization
Matplotlib
Plotly
Web Application
Streamlit
Model Persistence
Joblib
Development Tools
Jupyter Notebook
VS Code
Git & GitHub
📊 Dataset

The project is designed for the commonly used Credit Card Fraud Detection dataset.

Dataset Features
Time
V1
V2
V3
...
V28
Amount
Class
Target Variable
Class = 0 → Genuine Transaction
Class = 1 → Fraudulent Transaction

The dataset contains a strong class imbalance, making it suitable for demonstrating fraud-detection techniques.

⚠️ The project should be used with anonymized/educational transaction data. Never enter real credit-card numbers or sensitive financial information.


#⚖️ Handling Class Imbalance

One of the main challenges in this project is class imbalance.

A typical fraud dataset contains significantly more genuine transactions than fraudulent transactions.

Genuine Transactions
████████████████████████████████████████

Fraudulent Transactions
█

Training a model directly on such data can produce misleading results.

Therefore, the project uses:

1️⃣ SMOTE

SMOTE — Synthetic Minority Over-sampling Technique

SMOTE generates synthetic examples for the minority class.

Imbalanced Training Data
          ↓
        SMOTE
          ↓
Balanced Training Data
          ↓
    Model Training
2️⃣ Class Weighting

Classification models can also use:

class_weight="balanced"

to give greater importance to the minority fraud class.

🔒 SMOTE is applied only to the training data to avoid data leakage.

#🤖 Machine Learning Models
1. Logistic Regression

Used as a baseline classification algorithm.

Advantages
Simple
Fast
Easy to interpret
Supports probability prediction
Suitable for binary classification
2. Random Forest

An ensemble learning algorithm consisting of multiple decision trees.

Advantages
Handles nonlinear relationships
Captures complex feature interactions
Robust classification performance
Provides feature importance
Suitable for structured transaction data
📈 Model Evaluation

The project evaluates models using multiple metrics.

Metric	Purpose
Accuracy	Overall correctly classified transactions
Precision	Correctness of fraud predictions
Recall	Ability to detect actual fraud
F1-Score	Balance between precision and recall
ROC-AUC	Ability to distinguish the two classes
Confusion Matrix
                    Predicted
                 Genuine   Fraud
              ┌─────────┬─────────┐
Actual Genuine│   TN    │   FP    │
              ├─────────┼─────────┤
       Fraud  │   FN    │   TP    │
              └─────────┴─────────┘

Because fraud detection is an imbalanced classification problem, accuracy should not be considered by itself.

#🖥️ Streamlit Application

The project includes an interactive Streamlit web application.

#🏠 Dashboard

The dashboard displays:

Total transactions
Genuine transactions
Fraudulent transactions
Fraud percentage
Model performance
Transaction statistics
Interactive charts
🔍 Fraud Prediction

Users can enter transaction feature values and click:

🔍 Predict Transaction

The application returns:

┌─────────────────────────────────┐
│      🚨 FRAUDULENT TRANSACTION  │
├─────────────────────────────────┤
│ Prediction: Fraudulent           │
│                                 │
│ Fraud Probability: 96.84%       │
│ Genuine Probability: 3.16%      │
└─────────────────────────────────┘

or:

┌─────────────────────────────────┐
│       ✅ GENUINE TRANSACTION    │
├─────────────────────────────────┤
│ Prediction: Genuine              │
│                                 │
│ Fraud Probability: 2.31%       │
│ Genuine Probability: 97.69%    │
└─────────────────────────────────┘

The displayed probability represents the model's predicted probability, not a guaranteed real-world fraud risk.

📊 Analytics

The application provides interactive visualizations such as:

📊 Genuine vs Fraud distribution
💰 Transaction amount distribution
🚨 Fraud amount distribution
⏱️ Transaction time analysis
🔥 Correlation heatmap
📈 Fraud trends
🌳 Random Forest feature importance
🤖 Model Performance

The application provides a model comparison dashboard.

#Example structure:

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	—	—	—	—	—
Random Forest	—	—	—	—	—

The values are generated dynamically from the trained models and are not hard-coded.

🚀 Installation & Setup
1️⃣ Clone the Repository
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd credit-card-fraud-detection
2️⃣ Create Virtual Environment
Windows
python -m venv venv

Activate:

venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
python -m pip install -r requirements.txt
🧪 Train the Model

Run:

python src/train_model.py

The training process will:

Load Dataset
     ↓
Preprocess Data
     ↓
Split Data
     ↓
Apply SMOTE
     ↓
Train Models
     ↓
Evaluate Models
     ↓
Compare Models
     ↓
Save Best Model
▶️ Run the Streamlit Application

After training:

streamlit run app/streamlit_app.py

The application will open in your browser.

📸 Screenshots

Add screenshots of your application here after completing the project.

🏠 Dashboard
[ Add Dashboard Screenshot Here ]
🔍 Fraud Prediction
[ Add Prediction Screenshot Here ]
📊 Analytics
[ Add Analytics Screenshot Here ]
🤖 Model Performance
[ Add Model Performance Screenshot Here ]
🔐 Security & Privacy

This project is intended for educational and demonstration purposes.

❌ Do not enter real credit-card numbers.
❌ Do not use real customer financial information.
✅ Use anonymized or publicly available datasets.
✅ Treat model predictions as machine-learning outputs, not financial decisions.
⚠️ Limitations

The current system has several limitations:

Performance depends on the quality of the training dataset.
Historical transaction patterns may not represent future fraud patterns.
An anonymized dataset limits direct interpretation of individual features.
Model probability is not necessarily a calibrated financial risk score.
The application is an educational prototype rather than a production banking system.
Real-world fraud detection would require continuous monitoring and model updates.

#🔮 Future Enhancements

Possible future improvements include:

⚡ Real-time transaction monitoring
🌐 REST API integration
🗄️ Database integration
📧 Fraud alert notifications
🧠 XGBoost / LightGBM comparison
🔎 SHAP-based explainable AI
📊 Threshold optimization
☁️ Cloud deployment
🔐 Authentication and authorization
📈 Model monitoring
🔄 Continuous model retraining
🎓 Academic Information
Project Type

College Minor Project

Domain

Machine Learning / Artificial Intelligence / Data Analytics

Core Concepts
Binary Classification
        +
Imbalanced Data
        +
SMOTE
        +
Random Forest
        +
Logistic Regression
        +
Data Visualization
        +
Streamlit
📚 Learning Outcomes

Through this project, the following concepts are demonstrated:

Data preprocessing
Exploratory Data Analysis
Binary classification
Imbalanced dataset handling
SMOTE
Logistic Regression
Random Forest
Model evaluation
Confusion matrix
Precision and Recall
F1-score
ROC-AUC
Probability prediction
Feature importance
Streamlit application development
Git/GitHub project management

#👨‍💻 Author

Sri Harish

🎓 B.E. Computer Science Engineering
💻 Interested in Data Analytics, Machine Learning & Software Development

Connect
Ph.No: +91-9489231147
LinkedIn: Add your LinkedIn profile

#⭐ Support
If you find this project useful, consider giving the repository a ⭐ on GitHub📜 License

This project is intended for educational purposes. Add an appropriate open-source license if you plan to distribute the source code publicly.
