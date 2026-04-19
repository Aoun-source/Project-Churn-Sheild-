from setuptools import setup, find_packages

setup(
    name="churnshield",
    version="1.0.0",
    description="End-to-end Customer Churn Prediction Platform",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "shap>=0.44",
        "streamlit>=1.28",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "plotly>=5.0",
        "joblib>=1.3",
        "imbalanced-learn>=0.11",
        "rich>=13.0",
    ],
)
