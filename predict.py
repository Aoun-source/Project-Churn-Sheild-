"""
predict.py
----------
Loads a trained model and generates churn predictions for new customer data.
Each customer gets a probability score, a binary prediction, and a risk label.

Usage:
    python predict.py                              # Runs on 20-row demo
    python predict.py --input data/customers.csv
    python predict.py --input data/customers.csv --output results.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import BEST_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, THRESHOLD
from src.data_loader import load_raw_data
from src.feature_engineering import engineer_features
from src.preprocessing import clean_data


def predict_churn(
    input_path: str = None,
    threshold: float = THRESHOLD,
    output_path: str = None
) -> pd.DataFrame:

    if not BEST_MODEL_PATH.exists():
        print("No trained model found. Run: python train.py")
        sys.exit(1)

    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    print(f"Model loaded: {type(model).__name__}")

    if input_path:
        df = pd.read_csv(input_path)
        print(f"Input: {len(df):,} customers from {input_path}")
    else:
        print("No input file specified. Running on 20-row demo sample.")
        df = load_raw_data().head(20)

    customer_ids = df.get("customerID", pd.Series(range(len(df)), name="customerID"))

    df_eng = engineer_features(df)
    df_clean = clean_data(df_eng)

    true_labels = None
    if "Churn" in df_clean.columns:
        true_labels = df_clean["Churn"].values
        df_clean = df_clean.drop(columns=["Churn"])

    X = preprocessor.transform(df_clean)
    X_df = pd.DataFrame(X, columns=feature_names)

    churn_proba = model.predict_proba(X_df)[:, 1]
    churn_pred = (churn_proba >= threshold).astype(int)

    def risk_level(p):
        if p >= 0.70:   return "High"
        elif p >= 0.40: return "Medium"
        else:           return "Low"

    results = pd.DataFrame({
        "customerID": customer_ids.values,
        "churn_probability": churn_proba.round(4),
        "churn_prediction": np.where(churn_pred == 1, "Yes", "No"),
        "risk_level": [risk_level(p) for p in churn_proba],
    })

    if true_labels is not None:
        results["actual_churn"] = np.where(true_labels == 1, "Yes", "No")

    high  = (churn_proba >= 0.70).sum()
    med   = ((churn_proba >= 0.40) & (churn_proba < 0.70)).sum()
    low   = (churn_proba < 0.40).sum()

    print(f"\nPrediction Summary ({len(results):,} customers):")
    print(f"  High risk   : {high:,}  ({high/len(results)*100:.1f}%)")
    print(f"  Medium risk : {med:,}  ({med/len(results)*100:.1f}%)")
    print(f"  Low risk    : {low:,}  ({low/len(results)*100:.1f}%)")

    print("\nTop 5 highest-risk customers:")
    top5 = results.nlargest(5, "churn_probability")[
        ["customerID", "churn_probability", "risk_level"]
    ]
    print(top5.to_string(index=False))

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChurnShield Batch Predictor")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    results = predict_churn(
        input_path=args.input,
        threshold=args.threshold,
        output_path=args.output
    )

    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))
