import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load trained model
# -------------------------------
# Example: assuming you saved a model as model.pkl in the model/ folder
with open("model.ipynb", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is **Malignant (M)** or **Benign (B)** based on input features.")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    radius_mean = st.sidebar.slider("Radius Mean", 5.0, 30.0, 14.0)
    texture_mean = st.sidebar.slider("Texture Mean", 5.0, 40.0, 19.0)
    smoothness_mean = st.sidebar.slider("Smoothness Mean", 0.05, 0.2, 0.1)
    compactness_mean = st.sidebar.slider("Compactness Mean", 0.01, 0.3, 0.2)

    data = {
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -------------------------------
# Prediction
# -------------------------------
st.subheader("User Input Features")
st.write(input_df)

prediction = model.predict(input_df)
prediction_label = "Malignant (M)" if prediction[0] == 1 else "Benign (B)"

st.subheader("Prediction")
st.write(prediction_label)
