import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load trained model + features
# -------------------------------
with open("saved_models/knn_model.pkl", "rb") as f: 
    data = pickle.load(f)

model = data["model"]
features = data["features"]  # list of feature names used during training

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is **Malignant (M)** or **Benign (B)** based on input features.")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    # Collect inputs for the same features used in training
    # Example: just a subset shown here — expand to all features in your dataset
    radius_mean = st.sidebar.slider("Radius Mean", 5.0, 30.0, 14.0)
    texture_mean = st.sidebar.slider("Texture Mean", 5.0, 40.0, 19.0)
    smoothness_mean = st.sidebar.slider("Smoothness Mean", 0.05, 0.2, 0.1)
    compactness_mean = st.sidebar.slider("Compactness Mean", 0.01, 0.3, 0.2)

    # Build input dict with correct feature names
    data = {
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
    }

    # Create DataFrame with all training features
    # Missing features get filled with 0 or default values
    input_df = pd.DataFrame([data], columns=features).fillna(0)
    return input_df

input_df = user_input_features()

# -------------------------------
# Prediction
# -------------------------------
st.subheader("User Input Features")
st.write(input_df)

prediction = model.predict(input_df)
prediction_label = "Malignant (M)" if prediction[0] == "M" else "Benign (B)"

st.subheader("Prediction")
st.write(prediction_label)
