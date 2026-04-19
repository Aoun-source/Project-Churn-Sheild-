"""
03_Model_Training_Evaluation.py
--------------------------------
Trains all five models, evaluates them on the held-out test set,
generates ROC and precision-recall curves, and runs SHAP analysis
on the best model.

Usage:
    python notebooks/03_Model_Training_Evaluation.py
"""

# %% Imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_raw_data
from src.feature_engineering import engineer_features
from src.preprocessing import get_processed_data
from src.models import cross_validate_all, train_all_models
from src.evaluate import (
    evaluate_all_models, plot_roc_curves, plot_confusion_matrices,
    plot_metrics_comparison, find_optimal_threshold
)
from config import REPORTS_DIR

plt.rcParams["figure.dpi"] = 130

# %% Load and preprocess
print("Loading and preprocessing...")
df = load_raw_data()
df = engineer_features(df)
X_train, X_test, y_train, y_test, feature_names = get_processed_data(df)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}  |  Features: {len(feature_names)}")

# %% Cross-validation baseline
print("\nRunning cross-validation baseline...")
cv_results = cross_validate_all(X_train, y_train, cv=5)

# %% Train
print("\nTraining all models...")
models = train_all_models(X_train, y_train)

# %% Evaluate
print("\nEvaluating on test set...")
metrics_df = evaluate_all_models(models, X_test, y_test)

# %% ROC curves
plot_roc_curves(models, X_test, y_test, save=True)

# %% Confusion matrices
plot_confusion_matrices(models, X_test, y_test, save=True)

# %% Metrics comparison
plot_metrics_comparison(metrics_df, save=True)

# %% Best model and optimal threshold
best_name = metrics_df.iloc[0]["Model"]
best_model = models[best_name]
print(f"\nBest model: {best_name}")
opt_threshold = find_optimal_threshold(best_model, X_test, y_test)

# %% Precision-recall curves
from sklearn.metrics import precision_recall_curve, average_precision_score

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

for (name, model), color in zip(models.items(), colors):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=color, lw=2)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "09_precision_recall.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Plot 9 saved.")

# %% SHAP analysis
print(f"\nRunning SHAP analysis on {best_name}...")
try:
    from src.evaluate import shap_analysis
    shap_analysis(best_model, X_test, model_name=best_name)
except ImportError:
    print("SHAP not installed. Run: pip install shap")

# %% Summary
print(f"""
Model Training Summary
----------------------
Models trained    : 5 (LR, RF, XGBoost, LightGBM, MLP)
Best model        : {best_name}
Best ROC-AUC      : {metrics_df.iloc[0]['ROC-AUC']:.4f}
Optimal threshold : {opt_threshold:.3f}

Plots saved to reports/figures/
""")
