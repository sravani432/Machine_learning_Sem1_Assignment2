import pickle
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def train_and_save_naive_bayes(dataset_path="data/breast-cancer-wisconsin-data.csv", save_path="generated_models/naive_bayes.pkl"):
    df = pd.read_csv(dataset_path)
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved at {save_path}")
    return model, X_test, y_test

def load_naive_bayes(save_path="generated_models/naive_bayes.pkl"):
    with open(save_path, "rb") as f:
        return pickle.load(f)

def run_naive_bayes(dataset_path="data/breast-cancer-wisconsin-data.csv", save_path="generated_models/naive_bayes.pkl"):
    # Try loading existing model
    if os.path.exists(save_path):
        model = load_naive_bayes(save_path)
        df = pd.read_csv(dataset_path)
        le = LabelEncoder()
        df["diagnosis"] = le.fit_transform(df["diagnosis"])
        X = df.drop("diagnosis", axis=1)
        y = df["diagnosis"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        model, X_test, y_test = train_and_save_naive_bayes(dataset_path, save_path)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

    save_metrics("naive_bayes", metrics)
    return metrics
def save_metrics(model_name, metrics, file_path="results.json"):
    # Load existing results if file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Update with new model metrics
    results[model_name] = metrics

    # Save back to file
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
