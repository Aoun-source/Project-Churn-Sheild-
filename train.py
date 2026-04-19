"""
train.py
--------
Main entry point for the ChurnShield training pipeline.
Runs the full sequence: load data, engineer features, preprocess,
cross-validate, train models, evaluate, generate plots, and SHAP analysis.

Usage:
    python train.py              # Full pipeline with hyperparameter tuning
    python train.py --quick      # Skip tuning, use default hyperparameters
    python train.py --skip-shap  # Skip SHAP analysis (faster)
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import load_raw_data, validate_data
from src.feature_engineering import engineer_features
from src.preprocessing import get_processed_data
from src.models import cross_validate_all, tune_all_models, train_all_models
from src.evaluate import (
    evaluate_all_models, plot_roc_curves, plot_confusion_matrices,
    plot_metrics_comparison, shap_analysis, save_best_model,
    find_optimal_threshold
)


def main(quick: bool = False, skip_shap: bool = False, skip_plots: bool = False):

    total_start = time.time()

    print("\n" + "=" * 60)
    print("  ChurnShield — Customer Churn Prediction Pipeline")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/7] Loading and validating data...")
    df = load_raw_data()
    validate_data(df)

    # Step 2: Feature engineering
    print("\n[2/7] Engineering features...")
    df = engineer_features(df)

    # Step 3: Preprocess
    print("\n[3/7] Preprocessing (encode, scale, SMOTE)...")
    X_train, X_test, y_train, y_test, feature_names = get_processed_data(df)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Total features: {len(feature_names)}")

    # Step 4: Cross-validation
    print("\n[4/7] Running cross-validation baseline...")
    cv_results = cross_validate_all(X_train, y_train)

    # Step 5: Train or tune
    if quick:
        print("\n[5/7] Training models (quick mode, no tuning)...")
        models = train_all_models(X_train, y_train)
    else:
        print("\n[5/7] Tuning hyperparameters with GridSearchCV...")
        models = tune_all_models(X_train, y_train, quick_mode=False)

    # Step 6: Evaluate
    print("\n[6/7] Evaluating on test set...")
    metrics_df = evaluate_all_models(models, X_test, y_test)

    # Plots
    if not skip_plots:
        plot_roc_curves(models, X_test, y_test, save=True)
        plot_confusion_matrices(models, X_test, y_test, save=True)
        plot_metrics_comparison(metrics_df, save=True)

    # Save best model
    best_model, best_name = save_best_model(models, metrics_df)
    opt_threshold = find_optimal_threshold(best_model, X_test, y_test)

    # Step 7: SHAP
    if not skip_shap:
        print(f"\n[7/7] Running SHAP analysis on {best_name}...")
        shap_analysis(best_model, X_test, model_name=best_name)
    else:
        print("\n[7/7] SHAP analysis skipped.")

    elapsed = time.time() - total_start
    best = metrics_df.iloc[0]

    print("\n" + "=" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Best model  : {best['Model']}")
    print(f"  ROC-AUC     : {best['ROC-AUC']:.4f}")
    print(f"  F1-Score    : {best['F1-Score']:.4f}")
    print(f"  Accuracy    : {best['Accuracy']:.4f}")
    print(f"  Threshold   : {opt_threshold:.3f} (Youden optimized)")
    print("=" * 60)

    print("\nTo run the dashboard:")
    print("  streamlit run app/streamlit_app.py\n")

    return models, metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChurnShield Training Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Train with default params, skip grid search")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP explainability analysis")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip saving evaluation plots")
    args = parser.parse_args()

    main(
        quick=args.quick,
        skip_shap=args.skip_shap,
        skip_plots=args.skip_plots
    )
