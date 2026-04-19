"""
data_loader.py
--------------
Handles loading the Telco dataset from disk, or generating a synthetic
version if the file is not available. Also includes a basic validation
function that reports shape, missing values, and class balance.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_PATH, KAGGLE_DATASET, TARGET_COLUMN


def download_dataset():
    """
    Attempts to download the dataset using the Kaggle API.
    Requires ~/.kaggle/kaggle.json to be configured.
    """
    try:
        import kaggle
        print("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(RAW_DATA_PATH.parent),
            unzip=True
        )
        print(f"Dataset saved to {RAW_DATA_PATH}")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Download manually from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print(f"Place the file at: {RAW_DATA_PATH}")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw CSV. If the file is missing, tries Kaggle first,
    then falls back to synthetic data so the pipeline can still run.
    """
    if not path.exists():
        print(f"Dataset not found at {path}. Trying Kaggle download...")
        download_dataset()

    if not path.exists():
        print("Real dataset unavailable. Generating synthetic data for demo.")
        return generate_synthetic_data()

    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def generate_synthetic_data(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic Telco-like dataset that mirrors the statistical
    properties of the real Kaggle dataset. Useful for running the full
    pipeline without needing a Kaggle account.
    """
    rng = np.random.default_rng(random_state)
    n = n_samples

    tenure = rng.integers(0, 73, n)
    monthly_charges = rng.uniform(18, 119, n).round(2)
    total_charges = (tenure * monthly_charges * rng.uniform(0.9, 1.1, n)).round(2)

    churn_prob = (
        0.05
        + 0.003 * monthly_charges
        - 0.004 * tenure
        + rng.uniform(0, 0.1, n)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = rng.binomial(1, churn_prob, n)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.binomial(1, 0.16, n),
        "Partner": rng.choice(["Yes", "No"], n, p=[0.48, 0.52]),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.90, 0.10]),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22]),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n, p=[0.39, 0.39, 0.22]),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.33, 0.23, 0.22, 0.22]
        ),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": np.where(tenure == 0, rng.uniform(0, 50, n).round(2), total_charges),
        "Churn": np.where(churn == 1, "Yes", "No")
    })

    print(f"Synthetic dataset generated: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Churn rate: {df['Churn'].eq('Yes').mean():.1%}")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Prints a basic data quality report.
    """
    print("\nData Validation Report")
    print("-" * 45)
    print(f"Shape         : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Target column : {TARGET_COLUMN}")

    if TARGET_COLUMN in df.columns:
        print(f"Class counts  : {df[TARGET_COLUMN].value_counts().to_dict()}")
        churn_rate = df[TARGET_COLUMN].eq("Yes").mean()
        print(f"Churn rate    : {churn_rate:.2%}")
        if churn_rate < 0.25:
            print("  Note: Class imbalance detected. SMOTE will be applied.")

    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values:\n{missing[missing > 0]}")
    else:
        print("Missing values: none")

    dups = df.duplicated().sum()
    print(f"Duplicates    : {dups}")
    print("-" * 45)


if __name__ == "__main__":
    df = load_raw_data()
    validate_data(df)
    print(df.head(3).to_string())
