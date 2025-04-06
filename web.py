import streamlit as st
import numpy as np
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Heart Disease Prediction")
st.subheader("Enter Patient Data:")

age = st.slider("Age", 20, 80, 50)
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
thalach = st.slider("Max Heart Rate Achieved (thalach)", 70, 210, 150)
exang = st.radio("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
thal_2 = st.radio("Thal_2", [0, 1])
thal_3 = st.radio("Thal_3", [0, 1])
cp_0 = st.radio("Chest Pain Type 0 (cp_0)", [0, 1])
ca_0 = st.radio("Number of major vessels 0 (ca_0)", [0, 1])

input_data = np.array([[exang, age * chol, chol, trestbps, age, oldpeak,
                        thal_3, cp_0, ca_0, thalach, thal_2]])

input_data[:, [1, 2, 3, 9]] = scaler.transform(input_data[:, [1, 2, 3, 9]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High risk of heart disease with a {prob:.2f} chance.")
    else:
        st.success(f"No heart disease predicted with a {prob:.2f} chance.")
