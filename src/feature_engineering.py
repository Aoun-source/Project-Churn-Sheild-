"""
feature_engineering.py
-----------------------
Creates domain-driven features from the raw Telco columns.
The logic here comes from understanding the business: contracts,
pricing, tenure, and service bundles are all signals a telecom
company would actually use to assess churn risk.

New features added (13 total):
  - is_new_customer           tenure <= 6 months
  - is_long_term_customer     tenure >= 24 months
  - avg_monthly_spend         total charges / tenure
  - cumulative_spend_ratio    actual vs expected total spend
  - high_value_customer       monthly charges in top 25%
  - total_services            count of add-on services subscribed
  - charge_per_service        monthly charge divided by services used
  - has_security_bundle       online security AND tech support both active
  - has_streaming_bundle      streaming TV AND movies both active
  - contract_risk_score       encodes churn risk of contract type (1-3)
  - payment_risk_score        encodes churn risk of payment method (1-4)
  - internet_value_score      encodes internet service type (1-3)
  - total_risk_score          composite of the above risk scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 13 engineered features to the dataframe.
    The original columns are preserved; nothing is removed here.
    """
    df = df.copy()

    # Tenure segments
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["is_long_term_customer"] = (df["tenure"] >= 24).astype(int)

    # Avoid division by zero for any zero-tenure rows
    safe_tenure = df["tenure"].replace(0, 1)

    # Spend features
    df["avg_monthly_spend"] = df["TotalCharges"].apply(
        lambda x: pd.to_numeric(x, errors="coerce")
    ) / safe_tenure

    total_charges_num = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    expected_total = df["MonthlyCharges"] * df["tenure"]
    df["cumulative_spend_ratio"] = np.where(
        expected_total > 0,
        total_charges_num / expected_total.replace(0, 1),
        1.0
    )

    q75 = df["MonthlyCharges"].quantile(0.75)
    df["high_value_customer"] = (df["MonthlyCharges"] >= q75).astype(int)

    # Service count
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["total_services"] = sum(
        (df[col] == "Yes").astype(int)
        for col in service_cols if col in df.columns
    )

    df["charge_per_service"] = df["MonthlyCharges"] / df["total_services"].replace(0, 0.5)

    # Bundle flags
    if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
        df["has_security_bundle"] = (
            (df["OnlineSecurity"] == "Yes") & (df["TechSupport"] == "Yes")
        ).astype(int)

    if "StreamingTV" in df.columns and "StreamingMovies" in df.columns:
        df["has_streaming_bundle"] = (
            (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
        ).astype(int)

    # Risk scores
    contract_risk = {"Month-to-month": 3, "One year": 2, "Two year": 1}
    if "Contract" in df.columns:
        df["contract_risk_score"] = df["Contract"].map(contract_risk).fillna(2)

    payment_risk = {
        "Electronic check": 4,
        "Mailed check": 3,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 1
    }
    if "PaymentMethod" in df.columns:
        df["payment_risk_score"] = df["PaymentMethod"].map(payment_risk).fillna(3)

    internet_score = {"Fiber optic": 3, "DSL": 2, "No": 1}
    if "InternetService" in df.columns:
        df["internet_value_score"] = df["InternetService"].map(internet_score).fillna(1)

    df["total_risk_score"] = (
        df.get("contract_risk_score", 0) +
        df.get("payment_risk_score", 0) +
        df["is_new_customer"] * 2 +
        df.get("internet_value_score", 0)
    )

    print(f"Feature engineering complete. Columns: {df.shape[1]} (was {df.shape[1] - 13})")
    return df


if __name__ == "__main__":
    from data_loader import load_raw_data
    df = load_raw_data()
    df_eng = engineer_features(df)
    print(df_eng[["total_risk_score", "total_services", "is_new_customer"]].describe().round(2))
