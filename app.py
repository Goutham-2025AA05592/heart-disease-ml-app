import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    RocCurveDisplay
)

# ===============================
# Global Configuration
# ===============================
sns.set_theme(style="whitegrid")
st.set_page_config(
    page_title="Heart Disease Prediction | ML Benchmarking",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center; color:#d62828;'>🫀 Heart Disease Prediction – ML Model Evaluation</h1>",
    unsafe_allow_html=True
)
st.info(
    "Predict heart disease using multiple ML models. Upload your dataset, "
    "select a model, and evaluate performance with metrics, confusion matrix, and ROC curve."
)

# ===============================
# Paths & Artifacts
# ===============================
MODEL_DIR = Path("model")
DATA_DIR = Path("data")

# -------------------------------
# Cache models and scaler
# -------------------------------
@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.pkl")

@st.cache_resource
def load_feature_names():
    return joblib.load(MODEL_DIR / "feature_names.pkl")

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression_model.pkl"),
        "Decision Tree": joblib.load(MODEL_DIR / "decision_tree_model.pkl"),
        "KNN": joblib.load(MODEL_DIR / "knn_model.pkl"),
        "Naive Bayes": joblib.load(MODEL_DIR / "naive_bayes_model.pkl"),
        "Random Forest": joblib.load(MODEL_DIR / "random_forest_model.pkl"),
        "XGBoost": joblib.load(MODEL_DIR / "xgboost_model.pkl"),
    }

scaler = load_scaler()
feature_names = load_feature_names()
MODELS = load_models()

SAMPLE_DATA_PATH = DATA_DIR / "heart_disease_test_sample_500.csv"
sample_df = pd.read_csv(SAMPLE_DATA_PATH) if SAMPLE_DATA_PATH.exists() else None

# ===============================
# Session State for Benchmarking
# ===============================
if "benchmark" not in st.session_state:
    st.session_state.benchmark = {}

# ===============================
# Preprocessing Function
# ===============================
def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    for col in ["id", "dataset"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    y = (df["num"] > 0).astype(int)
    X = df.drop(columns=["num"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
        X[col] = X[col].infer_objects(copy=False)  # Prevent future warning

    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=feature_names, fill_value=0)
    return scaler.transform(X), y

# ===============================
# Sidebar
# ===============================
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select ML Model", list(MODELS.keys()))

if sample_df is not None:
    st.sidebar.download_button(
        "⬇ Download Sample Test Dataset",
        data=sample_df.to_csv(index=False),
        file_name="heart_disease_test_sample_500.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("📘 Assignment Info")
st.sidebar.markdown(
    """
    **Student ID:** 2025AA05592  
    **Program:** M.Tech (AIML)  
    **Course:** Machine Learning  
    **Institution:** BITS Pilani  
    """
)

# ===============================
# Main Tabs
# ===============================
tab1, tab2, tab3 = st.tabs(
    ["📊 Model Performance", "📈 Benchmarking", "📂 Data Exploration"]
)

# ===============================
# Landing State
# ===============================
if uploaded_file is None:
    st.info("👈 Upload a test dataset from the sidebar to start analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ===============================
# Run Model
# ===============================
with st.spinner("Analyzing dataset and running model..."):
    X_test, y_true = preprocess_data(df)
    model = MODELS[selected_model_name]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    st.session_state.benchmark[selected_model_name] = metrics

# ===============================
# TAB 1: Model Performance
# ===============================
with tab1:
    st.subheader(f"📊 {selected_model_name} Performance Metrics")

    colors = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93", "#ff6f91"]
    kpi_keys = list(metrics.keys())
    kpi_values = [f"{v:.4f}" for v in metrics.values()]

    cols = st.columns(6)
    for i, col in enumerate(cols):
        col.metric(label=kpi_keys[i], value=kpi_values[i])

    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("### ROC Curve")
    fig, ax = plt.subplots(figsize=(5,4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    st.pyplot(fig)

# ===============================
# TAB 2: Benchmarking
# ===============================
with tab2:
    st.subheader("📈 Model Benchmark Comparison")
    if st.session_state.benchmark:
        bench_df = pd.DataFrame(st.session_state.benchmark).T
        st.dataframe(bench_df.style.format("{:.4f}").highlight_max(axis=0, color="#8ac926"))
    else:
        st.info("Run at least one model to see benchmarking results.")

# ===============================
# TAB 3: Data Exploration
# ===============================
with tab3:
    st.subheader("🔍 Data Exploration")

    analysis_df = df.drop(columns=["id"], errors="ignore")

    st.markdown("### 📊 Numerical Features")
    num_df = analysis_df.select_dtypes(include=["int64", "float64"]).drop(columns=["num"], errors="ignore")
    if not num_df.empty:
        st.dataframe(num_df.describe())
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numerical features found.")

    st.markdown("### 🏷️ Categorical Features")
    cat_cols = analysis_df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            st.write(f"**{col}**")
            st.dataframe(analysis_df[col].value_counts().to_frame("Count"))
    else:
        st.info("No categorical features found.")

    st.markdown("### 🎯 Target Variable (`num`)")
    target_counts = analysis_df["num"].value_counts().to_frame("Count")
    target_proportion = analysis_df["num"].value_counts(normalize=True).to_frame("Proportion")
    st.dataframe(target_counts)
    st.dataframe(target_proportion)
