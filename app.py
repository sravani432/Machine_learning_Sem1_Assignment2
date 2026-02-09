import streamlit as st
from models.logistic_regression import run_logistic_regression
from models.decision_tree import run_decision_tree
from models.knn import run_knn
from models.naive_bayes import run_naive_bayes
from models.random_forest import run_random_forest
from models.xgboost_model import run_xgboost

st.title("Breast Cancer Classifier Comparison")

model_choice = st.selectbox(
    "Choose a model:",
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
    st.write(f"{k}: {v:.4f}")
