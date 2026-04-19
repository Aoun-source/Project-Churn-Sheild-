"""
preprocessing.py
----------------
Handles cleaning, encoding, scaling, and train/test splitting.
Produces a ColumnTransformer preprocessor that is saved to disk
so it can be reloaded at inference time without retraining.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    TARGET_COLUMN, ID_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    TEST_SIZE, RANDOM_STATE, SCALER, HANDLE_IMBALANCE,
    PREPROCESSOR_PATH, FEATURE_NAMES_PATH
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes data types, encodes binary columns, and drops the customer ID.
    Returns a cleaned DataFrame ready for the preprocessor.
    """
    df = df.copy()

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # TotalCharges has occasional whitespace values that cause parse issues
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode binary target
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # Encode gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # Encode simple Yes/No columns
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    return df


def build_preprocessor(scaler_type: str = SCALER) -> ColumnTransformer:
    """
    Returns a ColumnTransformer that scales numerics and one-hot encodes
    categorical features. Binary columns pass through unchanged.
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler()
    }
    scaler = scalers.get(scaler_type, StandardScaler())

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
             CATEGORICAL_FEATURES),
        ],
        remainder="passthrough"
    )
    return preprocessor


def preprocess(df: pd.DataFrame, fit: bool = True):
    """
    Runs the full preprocessing pipeline.

    If fit=True (training mode): fits the preprocessor, applies SMOTE,
    saves the preprocessor to disk, and returns train/test splits.

    If fit=False (inference mode): loads the saved preprocessor and
    transforms the input without refitting.
    """
    X = df.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in df.columns else df
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None

    if fit:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        preprocessor = build_preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        ohe_features = preprocessor.named_transformers_["cat"] \
            .get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        passthrough_features = [
            c for c in X.columns
            if c not in NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        ]
        feature_names = NUMERICAL_FEATURES + ohe_features + passthrough_features

        X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

        if HANDLE_IMBALANCE:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_df, y_train = smote.fit_resample(X_train_df, y_train)
            print(f"SMOTE applied: training set is now {len(y_train):,} samples "
                  f"({y_train.sum():,} churn / {(y_train == 0).sum():,} no churn)")

        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        joblib.dump(feature_names, FEATURE_NAMES_PATH)
        print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

        return (
            X_train_df,
            X_test_df,
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
            feature_names
        )

    else:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        X_proc = preprocessor.transform(X)
        return pd.DataFrame(X_proc, columns=feature_names)


def get_processed_data(df: pd.DataFrame):
    """
    Convenience wrapper: clean, then preprocess in training mode.
    Returns X_train, X_test, y_train, y_test, feature_names.
    """
    df_clean = clean_data(df)
    return preprocess(df_clean, fit=True)


if __name__ == "__main__":
    from data_loader import load_raw_data, validate_data

    df = load_raw_data()
    validate_data(df)
    X_train, X_test, y_train, y_test, feat_names = get_processed_data(df)

    print(f"\nTrain: X={X_train.shape}, y={y_train.shape}, churn rate={y_train.mean():.2%}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}, churn rate={y_test.mean():.2%}")
    print(f"Total features: {len(feat_names)}")
