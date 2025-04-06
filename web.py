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
sex = st.radio("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved (thalach)", 70, 210, 150)
exang = st.radio("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2])

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

input_data = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"High risk of heart disease with a {prob:.2f} chance.")
    else:
        st.success(f"No heart disease predicted with a {prob:.2f} chance.")
