# streamlit_diabetes_app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load trained pipeline
# -------------------------------
loaded_pipeline = pickle.load(open(
    r"C:\Users\lenovo\Desktop\github\diabetes classification using svm\diabetes_model.pkl", "rb"))

# Optional: feature names for importance plotting
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


# -------------------------------
# 2. Prediction function
# -------------------------------
def diabetes_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = loaded_pipeline.predict(input_array)
    probability = loaded_pipeline.predict_proba(input_array)[0][1]
    return prediction[0], probability


# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction Web App")
st.write("Enter patient health details below to predict diabetes.")

# Input fields
st.subheader("Patient Health Details")
Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
Glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
Insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=79)
BMI = st.number_input('BMI Value', min_value=0.0, max_value=70.0, value=32.0, step=0.1)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.47,
                                           step=0.01)
Age = st.number_input('Age of the Person', min_value=10, max_value=100, value=33)

# Prediction button
if st.button("Check Diabetes Status"):
    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                  Insulin, BMI, DiabetesPedigreeFunction, Age]

    prediction, probability = diabetes_prediction(input_data)

    # Display result
    if prediction == 0:
        st.success("✅ The person is **Non-Diabetic**")
    else:
        st.warning("⚠️ The person is **Diabetic**")

    # Show probability
    st.info(f"Probability of being diabetic: {probability * 100:.2f}%")

    # -------------------------------
    # Optional: Feature Importance for Random Forest / Tree models
    # -------------------------------
    if hasattr(loaded_pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = loaded_pipeline.named_steps['classifier'].feature_importances_
        sorted_idx = np.argsort(importances)
        plt.figure(figsize=(8, 5))
        plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='skyblue')
        plt.xlabel("Importance Score")
        plt.title("Feature Importance (Random Forest)")
        st.pyplot(plt)

