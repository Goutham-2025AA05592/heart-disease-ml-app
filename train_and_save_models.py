
"""
train_and_save_models.py
Train and save 6 ML models for Heart Disease Classification
Saves scaler, feature names, models, and metrics
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.logistic_regression_model import train_model as lr
from model.decision_tree_model import train_model as dt
from model.knn_model import train_model as knn
from model.naive_bayes_model import train_model as nb
from model.random_forest_model import train_model as rf
from model.xgboost_model import train_model as xgb

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("data/heart_disease_uci.csv")

# ===============================
# Binary Target Conversion
# ===============================
df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)
df.drop(columns=["num", "id", "dataset"], inplace=True)

# ===============================
# Separate Column Types
# ===============================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

# ===============================
# Handle Missing Values
# ===============================
# Numerical → median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical → mode
for col in categorical_cols:
    df[col] = (
        df[col]
        .fillna(df[col].mode()[0])
        .infer_objects(copy=False)
    )

# ===============================
# One-Hot Encoding
# ===============================
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ===============================
# Feature / Target Split
# ===============================
X = df.drop("target", axis=1)
y = df["target"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Feature Scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for Streamlit
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(X.columns.tolist(), "model/feature_names.pkl")

# ===============================
# Train & Save Models
# ===============================
models = {
    "logistic_regression_model.pkl": lr,
    "decision_tree_model.pkl": dt,
    "knn_model.pkl": knn,
    "naive_bayes_model.pkl": nb,
    "random_forest_model.pkl": rf,
    "xgboost_model.pkl": xgb
}

# ===============================
# Store metrics for all models
# ===============================
all_metrics = {}


for filename, trainer in models.items():
    model, metrics = trainer(X_train, X_test, y_train, y_test)
    joblib.dump(model, f"model/{filename}")

    print(f"\n{filename}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    all_metrics[filename] = metrics

pd.DataFrame(all_metrics).T.to_csv("model/model_metrics.csv")
