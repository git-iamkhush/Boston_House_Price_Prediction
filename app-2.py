import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Boston House Price Prediction", layout="centered")

st.title("🏠 Boston House Price Prediction")
st.write("Enter the house details below to predict the median house price.")

user_input = {}

for feature in feature_columns:
    user_input[feature] = st.number_input(
        feature,
        value=0.0,
        format="%.2f"
    )

input_df = pd.DataFrame([user_input])

if st.button("Predict House Price"):
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted House Price: {prediction:.2f}")