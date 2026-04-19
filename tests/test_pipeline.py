"""
test_pipeline.py
----------------
Unit and integration tests for ChurnShield.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_loader import generate_synthetic_data, validate_data
from src.feature_engineering import engineer_features
from src.preprocessing import clean_data


@pytest.fixture
def sample_df():
    return generate_synthetic_data(n_samples=300, random_state=42)


@pytest.fixture
def engineered_df(sample_df):
    return engineer_features(sample_df)


class TestDataLoader:

    def test_shape(self, sample_df):
        assert sample_df.shape[0] == 300
        assert sample_df.shape[1] == 21

    def test_required_columns(self, sample_df):
        for col in ["customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"]:
            assert col in sample_df.columns

    def test_churn_values(self, sample_df):
        assert set(sample_df["Churn"].unique()).issubset({"Yes", "No"})

    def test_tenure_range(self, sample_df):
        assert sample_df["tenure"].min() >= 0
        assert sample_df["tenure"].max() <= 72


class TestFeatureEngineering:

    def test_adds_columns(self, sample_df, engineered_df):
        assert engineered_df.shape[1] > sample_df.shape[1]

    def test_binary_flags(self, engineered_df):
        for col in ["is_new_customer", "is_long_term_customer"]:
            if col in engineered_df.columns:
                assert set(engineered_df[col].unique()).issubset({0, 1})

    def test_no_infinite_values(self, engineered_df):
        numeric = engineered_df.select_dtypes(include=np.number)
        assert not np.isinf(numeric).any().any()

    def test_no_new_nulls(self, sample_df, engineered_df):
        new_cols = [c for c in engineered_df.columns if c not in sample_df.columns]
        numeric_new = engineered_df[new_cols].select_dtypes(include=np.number)
        assert numeric_new.isnull().sum().sum() == 0

    def test_services_non_negative(self, engineered_df):
        if "total_services" in engineered_df.columns:
            assert (engineered_df["total_services"] >= 0).all()


class TestPreprocessing:

    def test_drops_customer_id(self, engineered_df):
        cleaned = clean_data(engineered_df)
        assert "customerID" not in cleaned.columns

    def test_encodes_target(self, engineered_df):
        cleaned = clean_data(engineered_df)
        assert cleaned["Churn"].isin([0, 1]).all()

    def test_total_charges_numeric(self, engineered_df):
        cleaned = clean_data(engineered_df)
        assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])


class TestIntegration:

    def test_full_preprocessing(self, sample_df):
        from src.preprocessing import get_processed_data
        df_eng = engineer_features(sample_df)
        X_train, X_test, y_train, y_test, feat_names = get_processed_data(df_eng)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.isin([0, 1]).all()

    def test_model_trains_and_predicts(self, sample_df):
        from sklearn.linear_model import LogisticRegression
        from src.preprocessing import get_processed_data
        df_eng = engineer_features(sample_df)
        X_train, X_test, y_train, y_test, _ = get_processed_data(df_eng)
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})
