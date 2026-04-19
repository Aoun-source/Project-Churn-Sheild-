# Dataset

## IBM Telco Customer Churn

Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

File to place here: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## Download Options

### Option A — Kaggle CLI

```bash
pip install kaggle
# Make sure ~/.kaggle/kaggle.json is configured with your API key

kaggle datasets download blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/
```

### Option B — Manual

1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click Download
3. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in this folder

### Option C — Skip it entirely

The pipeline detects if the file is missing and automatically generates
a synthetic dataset with the same statistical properties. You can run
`python train.py` right away without downloading anything.

---

## Column Reference

| Column | Type | Description |
|---|---|---|
| customerID | string | Unique customer ID |
| gender | categorical | Male / Female |
| SeniorCitizen | binary | 1 = senior |
| Partner | binary | Has partner |
| Dependents | binary | Has dependents |
| tenure | numeric | Months with company (0-72) |
| PhoneService | binary | Has phone service |
| MultipleLines | categorical | Yes / No / No phone service |
| InternetService | categorical | DSL / Fiber optic / No |
| OnlineSecurity | categorical | Yes / No / No internet service |
| OnlineBackup | categorical | Yes / No / No internet service |
| DeviceProtection | categorical | Yes / No / No internet service |
| TechSupport | categorical | Yes / No / No internet service |
| StreamingTV | categorical | Yes / No / No internet service |
| StreamingMovies | categorical | Yes / No / No internet service |
| Contract | categorical | Month-to-month / One year / Two year |
| PaperlessBilling | binary | Yes / No |
| PaymentMethod | categorical | 4 options including electronic check |
| MonthlyCharges | numeric | Monthly bill amount ($) |
| TotalCharges | numeric | Cumulative bill amount ($) |
| Churn | target | Yes / No |

Rows: 7,043  
Columns: 21  
Churn rate: ~26.5%
