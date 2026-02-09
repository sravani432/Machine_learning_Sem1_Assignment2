import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def run_logistic_regression(dataset_path="data/breast-cancer-wisconsin-data.csv",save_path="../saved_models/logistic_regression.pkl"):
    print("Running Logistic Regression...",dataset_path)
    df = pd.read_csv(dataset_path)
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # Save model 
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    with open(save_path, "wb") as f: 
        pickle.dump(model, f)
    print(f"Logistic Regression model saved to {save_path}")

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }
