https://fahnz6ep2icq5inth4pw3y.streamlit.app/
# ChurnShield — Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn in the telecom industry. Built to demonstrate a full ML workflow: from raw data and exploratory analysis, through feature engineering and model training, to a deployable prediction dashboard.

---

## Problem Statement

Customer churn is one of the most expensive problems in the telecom business. Studies show acquiring a new customer costs five to twenty-five times more than keeping an existing one. This project builds a complete pipeline to identify which customers are likely to leave, understand the reasons behind it, and estimate the financial impact.

The goal is not just to train a model, but to build something a business could actually use.

---

## Project Structure

```
ChurnShield/
|
|-- notebooks/
|   |-- 01_EDA_and_Visualization.py         Exploratory analysis, 9 plots
|   |-- 02_Feature_Engineering.py           Custom feature creation and selection
|   |-- 03_Model_Training_Evaluation.py     Training, tuning, and SHAP analysis
|
|-- src/
|   |-- data_loader.py          Data loading and synthetic data generation
|   |-- preprocessing.py        Cleaning, encoding, scaling, SMOTE
|   |-- feature_engineering.py  13 domain-driven engineered features
|   |-- models.py               5 model definitions and hyperparameter grids
|   |-- evaluate.py             Metrics, ROC curves, confusion matrices, SHAP
|
|-- app/
|   |-- streamlit_app.py        Interactive prediction dashboard
|
|-- data/
|   |-- README.md               Dataset download instructions
|
|-- models/                     Saved .pkl files after training
|-- reports/figures/            Auto-generated plots
|-- train.py                    Main pipeline entry point
|-- predict.py                  Batch prediction on new data
|-- config.py                   All paths and hyperparameters
|-- requirements.txt
```

---

## Models Used

Five algorithms are trained, cross-validated, and compared:

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 80.4% | 59.5% | 0.843 |
| Random Forest | 85.1% | 66.4% | 0.891 |
| XGBoost | 86.7% | 68.9% | 0.912 |
| LightGBM | 86.2% | 67.9% | 0.908 |
| Neural Network (MLP) | 84.3% | 65.3% | 0.878 |

Best model: XGBoost with a ROC-AUC of 0.912.

---

## Key Findings from Analysis

- Customers on month-to-month contracts churn at three times the rate of annual contract holders.
- Electronic check payment users show a churn rate of around 45%, compared to 15% for credit card users.
- Churn is highest in the first twelve months. Retaining customers in year one is critical.
- Higher monthly charges consistently correlate with higher churn probability.
- Customers without online security or tech support are twice as likely to leave.

---

## Top Predictive Features (SHAP)

```
Contract (Month-to-month)    0.42
Tenure                       0.31
Monthly Charges              0.28
Internet Service (Fiber)     0.24
Payment Method (E-Check)     0.21
Total Charges                0.19
Online Security              0.17
Tech Support                 0.15
Senior Citizen               0.12
Paperless Billing            0.10
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/syedaounhaider/ChurnShield.git
cd ChurnShield
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

The dataset is the IBM Telco Customer Churn dataset from Kaggle. Download instructions are in `data/README.md`. If you skip this step, the pipeline will automatically generate a realistic synthetic dataset so you can still run everything.

### 4. Run the training pipeline

```bash
python train.py
```

For a faster run without hyperparameter tuning:

```bash
python train.py --quick
```

### 5. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## The Dashboard

The Streamlit app has four sections:

- **Predict**: Enter a customer profile and get a real-time churn probability with risk level and recommended retention actions.
- **Data Explorer**: Browse the dataset, view distributions, and explore the correlation matrix.
- **Model Performance**: Compare all five models visually across every metric.
- **About**: Project overview, tech stack, and quick start guide.

---

## Tech Stack

- Python 3.9+
- Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM
- SHAP
- Streamlit
- Matplotlib, Seaborn, Plotly
- Imbalanced-learn (SMOTE)

---

## Future Improvements

- Add a FastAPI backend to serve predictions as a REST API
- Integrate MLflow for experiment tracking
- Deploy to Streamlit Cloud or Hugging Face Spaces
- Build a real-time monitoring dashboard for production drift detection

---

## Author

Syed Aoun Haider
Machine Learning Engineer
GitHub: github.com/syedaounhaider
