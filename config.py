"""
config.py
---------
Central configuration for ChurnShield.
All paths, column names, and hyperparameters are defined here.
Import this file in any module instead of hardcoding values.
"""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset
RAW_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
KAGGLE_DATASET = "blastchar/telco-customer-churn"

# Column definitions
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges"
]

BINARY_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling"
]

CATEGORICAL_FEATURES = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]

# Train / test split
RANDOM_STATE = 42
TEST_SIZE = 0.2
SCALER = "standard"
HANDLE_IMBALANCE = True

# Hyperparameter grids
LOGISTIC_REGRESSION_PARAMS = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [1000],
    "class_weight": ["balanced", None]
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "class_weight": ["balanced", None]
}

XGBOOST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1, 2, 3]
}

LIGHTGBM_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "class_weight": ["balanced", None]
}

MLP_PARAMS = {
    "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["adaptive"],
    "max_iter": [500]
}

# Evaluation
CV_FOLDS = 5
SCORING_METRIC = "roc_auc"
THRESHOLD = 0.5

# Figure settings
FIGURE_SIZE = (12, 8)
DPI = 150
CHURN_COLORS = {"Yes": "#e74c3c", "No": "#2ecc71"}

# Saved artifact paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
