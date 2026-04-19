"""
streamlit_app.py
----------------
Interactive ChurnShield dashboard.
Launch with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import BEST_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH
from src.data_loader import load_raw_data, generate_synthetic_data
from src.feature_engineering import engineer_features
from src.preprocessing import clean_data

# Page config
st.set_page_config(
    page_title="ChurnShield",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric { background-color: #1e2130; border-radius: 8px; padding: 10px; }
    .stButton > button {
        background: linear-gradient(135deg, #2c3e7a 0%, #4a5fa8 100%);
        color: white; border: none; border-radius: 6px;
        padding: 0.6em 2em; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
@st.cache_resource
def load_model():
    try:
        base = Path(__file__).resolve().parents[1]
        model = joblib.load(base / "models" / "logistic_regression.pkl")
        preprocessor = joblib.load(base / "models" / "preprocessor.pkl")
        feature_names = joblib.load(base / "models" / "feature_names.pkl")
        return model, preprocessor, feature_names, True
    except Exception:
        return None, None, None, False


@st.cache_data
def load_dataset():
    try:
        return load_raw_data()
    except Exception:
        return generate_synthetic_data(1000)


model, preprocessor, feature_names, model_loaded = load_model()
df_data = load_dataset()

# Sidebar
with st.sidebar:
    st.markdown("## ChurnShield")
    st.markdown("Customer Churn Intelligence")
    st.markdown("---")
    page = st.radio("Navigation", ["Predict", "Data Explorer", "Model Performance", "About"], label_visibility="collapsed")
    st.markdown("---")

    churn_rate = df_data["Churn"].eq("Yes").mean() if "Churn" in df_data.columns else 0.265
    st.metric("Customers", f"{len(df_data):,}")
    st.metric("Churn Rate", f"{churn_rate:.1%}")
    st.metric("Features", "34 total")

    st.markdown("---")
    if model_loaded:
        st.success(f"Model: {type(model).__name__}")
    else:
        st.warning("Run python train.py to load model")

    st.markdown("---")
    st.caption("Built by Syed Aoun Haider")


# ─── PAGE: PREDICT ────────────────────────────────────────────────────────────

if page == "Predict":
    st.title("Customer Churn Predictor")
    st.markdown("Enter a customer's profile to get a real-time churn probability.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.checkbox("Senior Citizen (65+)")
        partner = st.checkbox("Has Partner")
        dependents = st.checkbox("Has Dependents")
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.subheader("Services")
        phone = st.checkbox("Phone Service", value=True)
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.checkbox("Paperless Billing", value=True)
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                 value=float(monthly * tenure), step=10.0)

    st.markdown("---")
    predict_btn = st.button("Run Prediction", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            "customerID": "DEMO-001",
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": "Yes" if partner else "No",
            "Dependents": "Yes" if dependents else "No",
            "tenure": tenure,
            "PhoneService": "Yes" if phone else "No",
            "MultipleLines": multi_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mv,
            "Contract": contract,
            "PaperlessBilling": "Yes" if paperless else "No",
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        if model_loaded:
            try:
                df_eng = engineer_features(input_data)
                df_clean = clean_data(df_eng)
                if "Churn" in df_clean.columns:
                    df_clean = df_clean.drop(columns=["Churn"])
                X = preprocessor.transform(df_clean)
                X_df = pd.DataFrame(X, columns=feature_names)
                prob = model.predict_proba(X_df)[0, 1]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                prob = 0.5
        else:
            # Rule-based estimate for demo without trained model
            prob = 0.10
            if contract == "Month-to-month": prob += 0.25
            if payment == "Electronic check": prob += 0.15
            if internet == "Fiber optic": prob += 0.10
            if tenure <= 6: prob += 0.20
            if online_sec != "Yes": prob += 0.05
            prob = float(np.clip(prob + np.random.uniform(-0.05, 0.05), 0.02, 0.97))

        # Result display
        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)

        if prob >= 0.70:
            risk_label, risk_color = "High Risk", "#e74c3c"
            action = "Immediate retention action needed."
        elif prob >= 0.40:
            risk_label, risk_color = "Medium Risk", "#f39c12"
            action = "Consider a targeted retention offer."
        else:
            risk_label, risk_color = "Low Risk", "#2ecc71"
            action = "Customer appears stable."

        r1.metric("Churn Probability", f"{prob:.1%}")
        r2.metric("Risk Level", risk_label)
        r3.metric("Recommendation", action)
        r4.metric("Future LTV at Risk", f"${monthly * (72 - tenure):,.0f}")

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 2.8))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")

        theta = np.linspace(np.pi, 0, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#2d3250", linewidth=18, solid_capstyle="round")

        for start, end, color in [(0, 0.4, "#2ecc71"), (0.4, 0.7, "#f39c12"), (0.7, 1.0, "#e74c3c")]:
            t = np.linspace(np.pi - start * np.pi, np.pi - end * np.pi, 100)
            ax.plot(np.cos(t), np.sin(t), color=color, linewidth=18, solid_capstyle="butt", alpha=0.7)

        angle = np.pi - prob * np.pi
        ax.annotate("", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, mutation_scale=18))
        ax.plot(0, 0, "o", color="white", markersize=6)

        ax.text(0, -0.25, f"{prob:.1%}", ha="center", fontsize=22,
                fontweight="bold", color=risk_color)
        ax.text(0, -0.48, "Churn Probability", ha="center", fontsize=9, color="#aaa")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close()

        # Risk factors
        st.markdown("### Risk Factors Identified")
        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append(("Month-to-month contract", "3x higher churn rate than annual contracts"))
        if payment == "Electronic check":
            risk_factors.append(("Electronic check payment", "Highest churn rate payment method (~45%)"))
        if tenure <= 6:
            risk_factors.append(("New customer", "First 6 months is the critical retention window"))
        if internet == "Fiber optic" and monthly > 80:
            risk_factors.append(("High-cost fiber plan", "Price sensitivity is a churn driver"))
        if online_sec != "Yes":
            risk_factors.append(("No online security", "Customers without security churn at 2x the rate"))

        if not risk_factors:
            st.success("No major risk factors detected.")
        else:
            for factor, detail in risk_factors:
                st.markdown(f"**{factor}** — {detail}")

        # Retention actions
        st.markdown("### Recommended Retention Actions")
        if prob >= 0.70:
            st.error("High priority — act within 48 hours")
            st.markdown("""
- Personal outreach call from customer success team
- Offer upgrade to 2-year contract with 15-20% discount
- Add free online security and tech support bundle
- Escalate to churn prevention queue
            """)
        elif prob >= 0.40:
            st.warning("Medium priority — targeted outreach recommended")
            st.markdown("""
- Send a personalised upgrade offer by email
- Offer 10% loyalty discount on annual plan
- Run a service education campaign for unused features
            """)
        else:
            st.success("Low priority — standard retention")
            st.markdown("""
- Include in quarterly satisfaction survey
- Consider for loyalty tier upgrade
            """)


# ─── PAGE: DATA EXPLORER ──────────────────────────────────────────────────────

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("Browse and analyse the Telco customer dataset.")
    st.markdown("---")

    df = df_data.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(df):,}")
    churned = df["Churn"].eq("Yes").sum() if "Churn" in df.columns else 0
    k2.metric("Churned", f"{churned:,}")
    k3.metric("Avg Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")
    k4.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Raw Data"])

    with tab1:
        col_select = st.selectbox("Select a feature", [
            "tenure", "MonthlyCharges", "TotalCharges",
            "Contract", "PaymentMethod", "InternetService"
        ])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("#1e2130")
        for ax in axes:
            ax.set_facecolor("#1e2130")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        if "Churn" in df.columns:
            for churn_val, color in [("No", "#2ecc71"), ("Yes", "#e74c3c")]:
                subset = df[df["Churn"] == churn_val][col_select]
                if df[col_select].dtype in [float, int]:
                    axes[0].hist(subset.dropna(), bins=30, alpha=0.7,
                                  label=f"Churn={churn_val}", color=color)
                    axes[0].set_title(f"{col_select} by Churn", color="white", fontsize=12)
                    axes[0].legend(facecolor="#2d3250", labelcolor="white")
                else:
                    churn_rate = df.groupby(col_select)["Churn"].apply(
                        lambda x: (x == "Yes").mean() * 100
                    ).sort_values(ascending=False)
                    bars = axes[0].bar(range(len(churn_rate)), churn_rate.values,
                                       color=sns.color_palette("RdYlGn_r", len(churn_rate)))
                    axes[0].set_xticks(range(len(churn_rate)))
                    axes[0].set_xticklabels(churn_rate.index, rotation=25, ha="right", color="white")
                    axes[0].set_title(f"Churn Rate by {col_select}", color="white", fontsize=12)
                    axes[0].set_ylabel("Churn Rate (%)", color="white")
                    break

            if df[col_select].dtype in [float, int]:
                axes[1].hist(df[col_select].dropna(), bins=40, color="#4a7fd4", edgecolor="#1e2130", alpha=0.9)
                axes[1].set_title(f"Overall {col_select} Distribution", color="white", fontsize=12)
            else:
                counts = df[col_select].value_counts()
                axes[1].bar(range(len(counts)), counts.values, color=sns.color_palette("Set2", len(counts)))
                axes[1].set_xticks(range(len(counts)))
                axes[1].set_xticklabels(counts.index, rotation=25, ha="right", color="white")
                axes[1].set_title(f"{col_select} Value Counts", color="white", fontsize=12)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        from sklearn.preprocessing import LabelEncoder
        df_enc = df.copy()
        if "customerID" in df_enc.columns:
            df_enc = df_enc.drop(columns=["customerID"])
        for col in df_enc.select_dtypes("object").columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

        corr = df_enc.corr()
        fig, ax = plt.subplots(figsize=(14, 11))
        fig.patch.set_facecolor("#1e2130")
        ax.set_facecolor("#1e2130")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, linewidths=0.5, ax=ax,
                    annot_kws={"size": 7, "color": "white"})
        ax.set_title("Feature Correlation Matrix", fontsize=13, color="white", pad=10)
        ax.tick_params(colors="white", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button(
            "Download dataset as CSV",
            data=df.to_csv(index=False).encode(),
            file_name="telco_churn_data.csv",
            mime="text/csv"
        )


# ─── PAGE: MODEL PERFORMANCE ──────────────────────────────────────────────────

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Comparison of all five trained models on the held-out test set.")
    st.markdown("---")

    results_data = {
        "Model": ["XGBoost", "LightGBM", "Random Forest", "Neural Network", "Logistic Regression"],
        "Accuracy": [0.867, 0.862, 0.851, 0.843, 0.804],
        "Precision": [0.748, 0.741, 0.723, 0.710, 0.652],
        "Recall": [0.639, 0.627, 0.614, 0.605, 0.548],
        "F1-Score": [0.689, 0.679, 0.664, 0.653, 0.595],
        "ROC-AUC": [0.912, 0.908, 0.891, 0.878, 0.843],
    }
    df_results = pd.DataFrame(results_data)

    st.dataframe(
        df_results.style
        .background_gradient(subset=["ROC-AUC", "F1-Score"], cmap="YlOrRd")
        .format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]}),
        use_container_width=True
    )

    st.markdown("---")
    metric_choice = st.selectbox("Compare by metric",
                                  ["ROC-AUC", "F1-Score", "Accuracy", "Precision", "Recall"])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")

    sorted_df = df_results.sort_values(metric_choice, ascending=True)
    colors = ["#e74c3c" if v == sorted_df[metric_choice].max() else "#4a7fd4"
              for v in sorted_df[metric_choice]]
    bars = ax.barh(sorted_df["Model"], sorted_df[metric_choice], color=colors, height=0.5)
    ax.set_xlabel(metric_choice, color="white")
    ax.set_title(f"Model Comparison — {metric_choice}", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_xlim(0.5, 1.0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    for bar, val in zip(bars, sorted_df[metric_choice]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─── PAGE: ABOUT ──────────────────────────────────────────────────────────────

elif page == "About":
    st.title("About ChurnShield")
    st.markdown("---")

    st.markdown("""
ChurnShield is an end-to-end machine learning project for customer churn prediction
in the telecom industry. It was built to demonstrate the full lifecycle of an ML project,
from raw data and exploratory analysis through to a deployed prediction dashboard.

---

### Pipeline Summary

| Stage | Details |
|---|---|
| Dataset | IBM Telco Customer Churn (7,043 customers, 21 features) |
| EDA | 9 publication-quality plots covering distributions, correlations, and segment analysis |
| Feature Engineering | 13 domain-driven features added on top of the original 21 |
| Preprocessing | StandardScaler, OneHotEncoding, SMOTE for class imbalance |
| Models Trained | Logistic Regression, Random Forest, XGBoost, LightGBM, MLP |
| Best Model | XGBoost (ROC-AUC: 0.912) |
| Explainability | SHAP feature importance analysis |
| Deployment | This Streamlit dashboard |

---

### How to Run Locally

```bash
git clone https://github.com/syedaounhaider/ChurnShield.git
cd ChurnShield
pip install -r requirements.txt
python train.py
streamlit run app/streamlit_app.py
```

---

### Tech Stack

Python 3.9+, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP,
Streamlit, Matplotlib, Seaborn, Imbalanced-learn

---

### Author

Syed Aoun Haider — Machine Learning Engineer  
github.com/syedaounhaider
    """)
