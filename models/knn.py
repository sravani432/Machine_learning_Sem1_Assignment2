import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def train_and_save_knn_model(
    dataset_path="../data/breast-cancer-wisconsin-data.csv", 
    pkl_filename="../saved_models/knn_model.pkl"
):
    # Ensure the folder exists
    os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)

    # Load dataset from CSV
    df = pd.read_csv(dataset_path)

    # Separate features (X) and target (y)
    # Replace 'diagnosis' with the actual target column in your dataset
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    knn.fit(X_train, y_train)

    # Evaluate (optional)
    accuracy = knn.score(X_test, y_test)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save the trained model along with feature names
    with open(pkl_filename, "wb") as f:
        pickle.dump({"model": knn, "features": X.columns.tolist()}, f)

    print(f"Model saved to {pkl_filename}")

if __name__ == "__main__":
    train_and_save_knn_model()
