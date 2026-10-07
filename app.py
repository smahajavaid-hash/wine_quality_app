import streamlit as st
import pandas as pd
import joblib


# Load saved model
model_rf = joblib.load("wine_quality_score.pkl")


# Page title
st.title("🍷 Wine Quality Predictor")

st.write("Enter the wine details below to predict the quality score.")


# User inputs
fixed_acidity = st.number_input("Fixed Acidity", value=7.0)

volatile_acidity = st.number_input("Volatile Acidity", value=0.5)

citric_acid = st.number_input("Citric Acid", value=0.2)

residual_sugar = st.number_input("Residual Sugar", value=2.0)

chlorides = st.number_input("Chlorides", value=0.08)

free_sulfur_dioxide = st.number_input(
    "Free Sulfur Dioxide",
    value=15.0
)

total_sulfur_dioxide = st.number_input(
    "Total Sulfur Dioxide",
    value=40.0
)

density = st.number_input(
    "Density",
    value=0.996
)

ph = st.number_input(
    "pH",
    value=3.3
)

sulphates = st.number_input(
    "Sulphates",
    value=0.6
)

alcohol = st.number_input(
    "Alcohol",
    value=10.0
)


# Prediction button
if st.button("Predict Wine Quality"):

    user_data = pd.DataFrame({
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [residual_sugar],
        "chlorides": [chlorides],
        "free sulfur dioxide": [free_sulfur_dioxide],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "pH": [ph],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    })

    prediction = model_rf.predict(user_data)[0]

    st.success(
        f"Predicted Wine Quality: {prediction:.2f}"
    )