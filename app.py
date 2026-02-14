import pandas as pd
import streamlit as st
from models.logistic_regression import run_logistic_regression
from models.decision_tree import run_decision_tree
from models.knn import run_knn
from sklearn.model_selection import train_test_split
from models.naive_bayes import run_naive_bayes
from models.random_forest import run_random_forest
from models.xgboost_model import run_xgboost
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

st.title("Breast Cancer Classification Model Evaluation App")

# --- Dataset Download Option ---
sample_dataset_path = "data/test_data.csv"   # adjust path to your dataset
try:
    sample_df = pd.read_csv(sample_dataset_path)
    st.download_button(
        label="Download Sample Dataset",
        data=sample_df.to_csv(index=False),
        file_name="test_data.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning("Sample dataset not found at the specified path.")

# --- Dataset Upload ---
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())
    save_path="generated_models/decision_tree.pkl"

    # --- Simple assumption: last column is target ---
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_choice = st.selectbox(
        "Select a model:",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    if model_choice == "Logistic Regression":
        metrics = run_logistic_regression()
    elif model_choice == "Decision Tree":
        metrics = run_decision_tree()
    elif model_choice == "KNN":
        metrics = run_knn()
    elif model_choice == "Naive Bayes":
        metrics = run_naive_bayes()
    elif model_choice == "Random Forest":
        metrics = run_random_forest()
    elif model_choice == "XGBoost":
        metrics = run_xgboost()

    st.subheader(f"Results for {model_choice}")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):   # numeric values
            st.write(f"{k}: {v:.4f}")
        else:                             # non-numeric values (lists, dicts, etc.)
            continue
            #st.write(f"{k}: {v}")



    # --- Evaluation Metrics ---
    print(metrics)
    acc = metrics["Accuracy"]
    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {acc:.2f}")

    # --- Classification Report ---
    st.subheader("Classification Report")
    st.text(classification_report(metrics["y_test"], metrics["y_pred"]))
  
    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(metrics["y_test"], metrics["y_pred"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)





    
