"""
01_EDA_and_Visualization.py
---------------------------
Exploratory data analysis for the Telco churn dataset.
Run directly as a Python script, or open in Jupyter as a notebook.
All plots are saved to reports/figures/.

Usage:
    python notebooks/01_EDA_and_Visualization.py
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

from src.data_loader import load_raw_data, validate_data
from config import REPORTS_DIR, CHURN_COLORS

plt.rcParams["figure.dpi"] = 130
sns.set_style("whitegrid")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
print("Setup done.")

# %% [markdown]
# ## 1. Load and Validate Data

# %%
df = load_raw_data()
validate_data(df)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# %% [markdown]
# ## 2. Churn Distribution
#
# The dataset is imbalanced: about 26.5% of customers churned.
# This matters for model training — we'll address it with SMOTE later.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

churn_counts = df["Churn"].value_counts()
colors = [CHURN_COLORS["No"], CHURN_COLORS["Yes"]]

bars = axes[0].bar(churn_counts.index, churn_counts.values,
                    color=colors, edgecolor="white", linewidth=2, width=0.5)
for bar, val in zip(bars, churn_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 40,
                 f"{val:,}  ({val/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[0].set_title("Churn Count", fontsize=13, fontweight="bold")
axes[0].set_ylim(0, churn_counts.max() * 1.2)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].pie(churn_counts.values, labels=["No Churn", "Churn"],
             autopct="%1.1f%%", colors=colors, startangle=90,
             wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
             textprops={"fontsize": 12})
axes[1].set_title("Churn Proportion", fontsize=13, fontweight="bold")

plt.suptitle("26.5% of customers churned — class imbalance detected",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "01_churn_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Plot 1 saved.")

# %% [markdown]
# ## 3. Numerical Feature Distributions
#
# Tenure, monthly charges, and total charges are the three continuous features.
# The distributions look different between churned and retained customers.

# %%
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for i, col in enumerate(num_cols):
    for churn_val, color in CHURN_COLORS.items():
        subset = df[df["Churn"] == churn_val][col]
        axes[0, i].hist(subset, bins=30, alpha=0.7, label=f"Churn={churn_val}",
                        color=color, edgecolor="white")
    axes[0, i].set_title(f"{col} Distribution", fontsize=11, fontweight="bold")
    axes[0, i].set_xlabel(col)
    axes[0, i].legend()
    axes[0, i].spines["top"].set_visible(False)
    axes[0, i].spines["right"].set_visible(False)

    bp = axes[1, i].boxplot(
        [df[df["Churn"] == "No"][col].values, df[df["Churn"] == "Yes"][col].values],
        labels=["No Churn", "Churn"], patch_artist=True, notch=True,
        medianprops=dict(color="black", linewidth=2)
    )
    for patch, color in zip(bp["boxes"], [CHURN_COLORS["No"], CHURN_COLORS["Yes"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[1, i].set_title(f"{col} by Churn", fontsize=11, fontweight="bold")
    axes[1, i].spines["top"].set_visible(False)
    axes[1, i].spines["right"].set_visible(False)

plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "02_numerical_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Plot 2 saved.")

# %% [markdown]
# ## 4. Churn Rate by Categorical Features
#
# Contract type and payment method are among the strongest signals.

# %%
cat_cols = ["Contract", "PaymentMethod", "InternetService",
            "SeniorCitizen", "Partner", "Dependents"]

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    churn_rate = df.groupby(col)["Churn"].apply(
        lambda x: (x == "Yes").sum() / len(x) * 100
    ).reset_index()
    churn_rate.columns = [col, "Churn Rate (%)"]
    churn_rate = churn_rate.sort_values("Churn Rate (%)", ascending=False)

    bars = axes[i].bar(range(len(churn_rate)), churn_rate["Churn Rate (%)"],
                        color=sns.color_palette("RdYlGn_r", len(churn_rate)),
                        edgecolor="white", linewidth=1.5)
    axes[i].set_xticks(range(len(churn_rate)))
    axes[i].set_xticklabels(churn_rate[col].astype(str), rotation=20, ha="right", fontsize=9)
    axes[i].set_title(f"Churn Rate by {col}", fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Churn Rate (%)")
    axes[i].axhline(y=df["Churn"].eq("Yes").mean() * 100,
                    color="navy", linestyle="--", alpha=0.5, label="Overall average")
    axes[i].legend(fontsize=8)
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

    for bar, val in zip(bars, churn_rate["Churn Rate (%)"]):
        axes[i].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.4,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.suptitle("Churn Rate by Categorical Features", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "03_categorical_churn_rates.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Plot 3 saved.")

# %% [markdown]
# ## 5. Correlation Heatmap

# %%
from sklearn.preprocessing import LabelEncoder
df_enc = df.copy()
df_enc["Churn_bin"] = df_enc["Churn"].map({"Yes": 1, "No": 0})
for col in df_enc.select_dtypes("object").columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
if "customerID" in df_enc.columns:
    df_enc = df_enc.drop(columns=["customerID"])

corr = df_enc.corr()
fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, annot_kws={"size": 7.5}, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Plot 4 saved.")

# %% [markdown]
# ## 6. Tenure vs Churn
#
# Churn is heavily concentrated in the first year.
# Customers who make it past month 24 have very low churn rates.

# %%
df["tenure_group"] = pd.cut(df["tenure"], bins=[0, 6, 12, 24, 36, 60, 72],
                             labels=["0-6m", "7-12m", "13-24m", "25-36m", "37-60m", "61-72m"])
churn_by_tenure = df.groupby("tenure_group")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(churn_by_tenure.index, churn_by_tenure.values,
                    color=plt.cm.RdYlGn_r(churn_by_tenure.values / 100), edgecolor="white")
for bar, val in zip(bars, churn_by_tenure.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[0].set_title("Churn Rate by Tenure Group", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Tenure Group")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].scatter(df[df["Churn"]=="No"]["tenure"], df[df["Churn"]=="No"]["MonthlyCharges"],
                alpha=0.15, color=CHURN_COLORS["No"], s=14, label="No Churn")
axes[1].scatter(df[df["Churn"]=="Yes"]["tenure"], df[df["Churn"]=="Yes"]["MonthlyCharges"],
                alpha=0.3, color=CHURN_COLORS["Yes"], s=14, label="Churn")
axes[1].set_title("Tenure vs Monthly Charges", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Tenure (months)")
axes[1].set_ylabel("Monthly Charges ($)")
axes[1].legend()
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("First year is the highest-risk period for churn", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "05_tenure_churn.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Plot 5 saved.")

# %% [markdown]
# ## 7. Service Adoption Heatmap

# %%
service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]

heatmap_data = {}
for col in service_cols:
    if col in df.columns:
        rates = df.groupby("Churn")[col].apply(lambda x: (x == "Yes").mean() * 100)
        heatmap_data[col] = rates

heatmap_df = pd.DataFrame(heatmap_data).T
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Adoption Rate (%)"})
ax.set_title("Service Adoption Rate by Churn Status (%)", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "06_service_adoption.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Plot 6 saved.")

# %% [markdown]
# ## 8. Key Takeaways

# %%
print("""
EDA Summary
-----------
1.  Overall churn rate is 26.5% — class imbalance, SMOTE will be used.
2.  Month-to-month contract customers churn at 3x the rate of annual holders.
3.  Churn is highest in the first 12 months of customer tenure.
4.  Electronic check payment has the highest churn rate (~45%).
5.  Fiber optic internet users churn more than DSL users despite faster speeds.
6.  Senior citizens churn at 41% vs 23% for non-seniors.
7.  Customers without online security churn at nearly double the rate.
8.  Higher monthly charges consistently predict higher churn probability.

All plots saved to reports/figures/
""")
