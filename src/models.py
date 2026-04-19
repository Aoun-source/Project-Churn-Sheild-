"""
models.py
---------
Defines and trains the five models used in ChurnShield.
Supports both a quick training mode (default hyperparameters)
and a full tuning mode (GridSearchCV with stratified cross-validation).
Each trained model is saved as a .pkl file in the models/ directory.
"""

import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    RANDOM_STATE, CV_FOLDS, SCORING_METRIC, MODELS_DIR,
    LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, MLP_PARAMS
)


def get_models() -> Dict[str, Any]:
    """Returns all five models with their base configurations."""
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "Neural Network": MLPClassifier(random_state=RANDOM_STATE, max_iter=500),
    }


def get_param_grids() -> Dict[str, Dict]:
    """Returns hyperparameter grids for each model."""
    return {
        "Logistic Regression": LOGISTIC_REGRESSION_PARAMS,
        "Random Forest": RANDOM_FOREST_PARAMS,
        "XGBoost": XGBOOST_PARAMS,
        "LightGBM": LIGHTGBM_PARAMS,
        "Neural Network": MLP_PARAMS,
    }


def cross_validate_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = CV_FOLDS
) -> pd.DataFrame:
    """
    Runs 5-fold stratified cross-validation on all five models.
    Useful for getting a baseline before tuning.
    Returns a DataFrame sorted by mean ROC-AUC.
    """
    print(f"\nCross-Validation Baseline ({cv}-Fold Stratified)")
    print("-" * 65)

    models = get_models()
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in models.items():
        start = time.time()
        roc_scores = cross_val_score(model, X_train, y_train,
                                     cv=cv_strategy, scoring="roc_auc", n_jobs=-1)
        f1_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv_strategy, scoring="f1", n_jobs=-1)
        elapsed = time.time() - start

        results.append({
            "Model": name,
            "ROC-AUC": f"{roc_scores.mean():.4f} +/- {roc_scores.std():.4f}",
            "F1-Score": f"{f1_scores.mean():.4f} +/- {f1_scores.std():.4f}",
            "ROC_mean": roc_scores.mean(),
            "Time": f"{elapsed:.1f}s"
        })
        print(f"  {name:<22}  ROC-AUC: {roc_scores.mean():.4f} +/- {roc_scores.std():.4f}  [{elapsed:.1f}s]")

    print("-" * 65)
    return pd.DataFrame(results).sort_values("ROC_mean", ascending=False)


def tune_model(
    name: str,
    model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3
) -> Tuple[Any, dict]:
    """
    Runs GridSearchCV for a single model and returns the best estimator.
    Uses 3-fold CV during tuning to keep runtime manageable.
    """
    print(f"\nTuning {name}...")
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=SCORING_METRIC,
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"  Best {SCORING_METRIC}: {grid_search.best_score_:.4f}")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Completed in {elapsed:.1f}s")

    return grid_search.best_estimator_, grid_search.best_params_


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quick_mode: bool = False
) -> Dict[str, Any]:
    """
    Tunes all five models and saves each to disk.
    Set quick_mode=True to use smaller grids for faster iteration.
    """
    print("\nHyperparameter Tuning")
    print("=" * 65)

    models = get_models()
    param_grids = get_param_grids()

    if quick_mode:
        param_grids["Logistic Regression"] = {"C": [0.1, 1.0, 10.0]}
        param_grids["Random Forest"] = {"n_estimators": [100, 200], "max_depth": [5, 10]}
        param_grids["XGBoost"] = {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
        param_grids["LightGBM"] = {"n_estimators": [100, 200], "max_depth": [5, 7], "learning_rate": [0.05, 0.1]}
        param_grids["Neural Network"] = {"hidden_layer_sizes": [(64,), (128, 64)], "alpha": [0.001]}

    tuned_models = {}

    for name, model in models.items():
        best_model, _ = tune_model(name, model, param_grids[name], X_train, y_train)
        tuned_models[name] = best_model

        model_path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(best_model, model_path)

    print(f"\nAll {len(tuned_models)} tuned models saved to {MODELS_DIR}/")
    return tuned_models


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Any]:
    """
    Trains all models with default parameters. No grid search.
    Use this for quick runs, testing, or debugging.
    """
    print("\nTraining all models (default hyperparameters)")
    print("-" * 50)

    models = get_models()
    trained = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        trained[name] = model
        print(f"  {name:<22} done in {elapsed:.1f}s")

        model_path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_path)

    return trained
