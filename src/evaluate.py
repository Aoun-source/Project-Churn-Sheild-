"""
evaluate.py
-----------
Handles all model evaluation: computing metrics, generating plots,
finding optimal classification thresholds, and running SHAP analysis.
All plots are saved to reports/figures/ automatically.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import REPORTS_DIR, BEST_MODEL_PATH, THRESHOLD

warnings.filterwarnings("ignore")


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = THRESHOLD
) -> dict:
    """
    Computes all classification metrics for a single model.
    Returns a dictionary with Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Avg Precision": average_precision_score(y_test, y_prob),
    }


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluates all models and prints a comparison table.
    Returns a DataFrame sorted by ROC-AUC descending.
    """
    print("\nModel Evaluation Results")
    print("=" * 85)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>9} {'Recall':>7} "
          f"{'F1':>7} {'ROC-AUC':>9}")
    print("-" * 85)

    all_metrics = []
    for name, model in models.items():
        m = evaluate_model(model, X_test, y_test, model_name=name)
        all_metrics.append(m)
        print(f"  {name:<22} {m['Accuracy']:>9.4f} {m['Precision']:>9.4f} "
              f"{m['Recall']:>7.4f} {m['F1-Score']:>7.4f} {m['ROC-AUC']:>9.4f}")

    print("=" * 85)
    df = pd.DataFrame(all_metrics).sort_values("ROC-AUC", ascending=False)
    print(f"\nBest model: {df.iloc[0]['Model']}  (ROC-AUC: {df.iloc[0]['ROC-AUC']:.4f})")
    return df


def find_optimal_threshold(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Finds the classification threshold that maximises Youden's J statistic
    (sensitivity + specificity - 1). More useful than the default 0.5 when
    the dataset has class imbalance.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    youdens_j = tpr - fpr
    best_idx = np.argmax(youdens_j)
    best_threshold = thresholds[best_idx]
    print(f"Optimal threshold (Youden's J): {best_threshold:.3f} "
          f"  TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f}")
    return best_threshold


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True
):
    """Plots ROC curves for all five models on a single chart."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2.5)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_title("ROC Curves", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        path = REPORTS_DIR / "roc_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True
):
    """Plots confusion matrices for all models in a grid."""
    n = len(models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.grid(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        path = REPORTS_DIR / "confusion_matrices.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame, save: bool = True):
    """Bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC across all models."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_melted = metrics_df.melt(id_vars="Model", value_vars=metric_cols,
                                 var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model",
                palette=sns.color_palette("Set2", len(metrics_df)),
                ax=ax, edgecolor="white", linewidth=0.8)

    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim([0.5, 1.0])
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        path = REPORTS_DIR / "metrics_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def shap_analysis(
    model,
    X_test: pd.DataFrame,
    model_name: str = "Best Model",
    max_samples: int = 500,
    save: bool = True
):
    """
    Runs SHAP on the best model.
    Generates two plots: a beeswarm summary and a mean |SHAP| bar chart.
    Both are saved to reports/figures/.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None

    print(f"\nRunning SHAP analysis on {model_name}...")
    X_sample = X_test.sample(min(max_samples, len(X_test)), random_state=42)

    model_type = type(model).__name__
    if any(t in model_type for t in ["XGB", "LGBM", "Forest"]):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample.head(50))

    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, show=False)
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        path = REPORTS_DIR / "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()

    # Mean |SHAP| bar chart
    mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=X_sample.columns)
    mean_shap = mean_shap.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(mean_shap.index, mean_shap.values,
            color=plt.cm.RdYlGn_r(mean_shap.values / mean_shap.values.max()),
            edgecolor="white")
    ax.set_title(f"Top 20 Features by Mean |SHAP| — {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        path = REPORTS_DIR / "shap_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    plt.close()

    print("\nTop 10 features by SHAP importance:")
    for i, (feat, val) in enumerate(mean_shap.sort_values(ascending=False).head(10).items(), 1):
        print(f"  {i:2}. {feat:<35} {val:.4f}")

    return mean_shap.sort_values(ascending=False)


def save_best_model(models: Dict, metrics_df: pd.DataFrame):
    """Saves the best performing model (by ROC-AUC) to models/best_model.pkl."""
    best_name = metrics_df.iloc[0]["Model"]
    best_model = models[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\nBest model '{best_name}' saved to {BEST_MODEL_PATH}")
    return best_model, best_name
