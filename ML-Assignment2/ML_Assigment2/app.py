"""
Streamlit App - ML Classification Model Comparison
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository

Features:
    - Upload test data (CSV)
    - Select from 5 classification models
    - View evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
    - Confusion Matrix and Classification Report
    - Compare all models side by side
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# --- Page configuration ---
st.set_page_config(
    page_title="ML Classification Models - Assignment 2",
    page_icon="🔬",
    layout="wide",
)

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_MAP = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

CLASS_NAMES = ["Malignant (0)", "Benign (1)"]


# --- Helper functions ---
@st.cache_resource
def load_model(model_file):
    """Load a saved model from the model directory."""
    path = os.path.join(MODEL_DIR, model_file)
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    """Load the saved StandardScaler."""
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    return joblib.load(path)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all 6 evaluation metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred):
    """Create a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    return fig


def plot_metrics_bar(metrics_dict):
    """Create a bar chart of metrics for a single model."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Evaluation Metrics", fontsize=14)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    return fig


def plot_comparison_chart(results_df):
    """Create a grouped bar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results_df.columns))
    width = 0.15
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for i, (model_name, row) in enumerate(results_df.iterrows()):
        offset = (i - len(results_df) / 2 + 0.5) * width
        ax.bar(x + offset, row.values, width, label=model_name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(results_df.columns, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison Across All Metrics", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    return fig


# --- Main App ---
st.title("ML Classification Model Comparison")
st.markdown(
    """
    **Dataset:** Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository  
    **Task:** Binary Classification (Malignant vs Benign)  
    **Models:** Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest
    """
)

st.divider()

# --- Sidebar ---
st.sidebar.header("Configuration")

# CSV upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"],
    help="Upload a CSV file with features and a 'target' column. "
    "If not uploaded, default test data is used.",
)

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(MODEL_MAP.keys()),
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **About this app**  
    Built for BITS MTech AIML - ML Assignment 2.  
    Compares 5 classification models on the 
    Breast Cancer Wisconsin dataset.
    
    **Author:** Sushil Kumar  
    **BITS ID:** 2025ac05637
    """
)

# --- Load data ---
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Custom test data uploaded successfully.")
else:
    default_path = os.path.join(BASE_DIR, "test_data.csv")
    data = pd.read_csv(default_path)
    st.info("Using default test data (114 samples). Upload your own CSV from the sidebar.")

# Validate data
if "target" not in data.columns:
    st.error(
        "The uploaded CSV must contain a 'target' column with class labels (0 or 1)."
    )
    st.stop()

X_data = data.drop(columns=["target"])
y_data = data["target"]

# Load scaler
scaler = load_scaler()

# Check feature count
expected_features = 30
if X_data.shape[1] != expected_features:
    st.error(
        f"Expected {expected_features} features, but got {X_data.shape[1]}. "
        "Make sure the CSV has the correct columns."
    )
    st.stop()

# Scale features
X_scaled = scaler.transform(X_data)

# --- Selected Model Results ---
st.header(f"Results: {selected_model}")

model = load_model(MODEL_MAP[selected_model])
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]
metrics = compute_metrics(y_data, y_pred, y_prob)

# Metric cards
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
with col2:
    st.metric("AUC Score", f"{metrics['AUC']:.4f}")
with col3:
    st.metric("Precision", f"{metrics['Precision']:.4f}")
with col4:
    st.metric("Recall", f"{metrics['Recall']:.4f}")
with col5:
    st.metric("F1 Score", f"{metrics['F1']:.4f}")
with col6:
    st.metric("MCC Score", f"{metrics['MCC']:.4f}")

# Confusion matrix and metrics bar chart side by side
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Confusion Matrix")
    fig_cm = plot_confusion_matrix(y_data, y_pred)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with chart_col2:
    st.subheader("Metrics Visualization")
    fig_bar = plot_metrics_bar(metrics)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

# Classification report
st.subheader("Classification Report")
report = classification_report(y_data, y_pred, target_names=CLASS_NAMES)
st.code(report, language="text")

# --- All Models Comparison ---
st.divider()
st.header("All Models Comparison")

all_results = {}
for name, file in MODEL_MAP.items():
    m = load_model(file)
    pred = m.predict(X_scaled)
    prob = m.predict_proba(X_scaled)[:, 1]
    all_results[name] = compute_metrics(y_data, pred, prob)

results_df = pd.DataFrame(all_results).T
results_df = results_df.round(4)

# Styled comparison table
st.subheader("Metrics Comparison Table")
st.dataframe(
    results_df.style.highlight_max(axis=0, color="#90EE90")
    .highlight_min(axis=0, color="#FFB6C1")
    .format("{:.4f}"),
    use_container_width=True,
)

# Grouped bar chart
st.subheader("Visual Comparison")
fig_comp = plot_comparison_chart(results_df)
st.pyplot(fig_comp)
plt.close(fig_comp)

# Best model summary
st.subheader("Best Model Per Metric")
best_col1, best_col2, best_col3 = st.columns(3)
metric_list = list(results_df.columns)
cols = [best_col1, best_col2, best_col3]
for i, metric in enumerate(metric_list):
    with cols[i % 3]:
        best_model = results_df[metric].idxmax()
        best_val = results_df.loc[best_model, metric]
        st.markdown(f"**Best {metric}**")
        st.markdown(f"{best_model} — `{best_val:.4f}`")

# Overall winner
ranks = results_df.rank(ascending=False)
avg_rank = ranks.mean(axis=1)
winner = avg_rank.idxmin()
st.success(f"**Overall Best Model (by average rank across all metrics): {winner}**")

# --- Dataset info ---
st.divider()
st.header("Dataset Information")
st.markdown(
    """
    **Name:** Breast Cancer Wisconsin (Diagnostic)  
    **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
    **Instances:** 569 (455 train / 114 test)  
    **Features:** 30 numeric features computed from digitized images of fine needle aspirates (FNA) of breast masses  
    **Classes:** Malignant (0) and Benign (1)  
    **Feature Categories:** Mean, standard error, and worst (largest) values of 10 cell nucleus properties:
    radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
    """
)

with st.expander("View Test Data Sample"):
    st.dataframe(data.head(20), use_container_width=True)
