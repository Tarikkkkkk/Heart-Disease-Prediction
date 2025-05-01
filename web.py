import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model.pkl')
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("Heart Disease Prediction App")
st.write("Input patient data below to assess heart disease risk.")
st.markdown("""
**Note:**
- `Sex`: 0 = Female, 1 = Male  
- `Fasting Blood Sugar (fbs)`: 1 = > 120 mg/dl, 0 = ≤ 120 mg/dl  
- `Exercise Induced Angina (exang)`: 1 = Yes, 0 = No  
""")

age = st.number_input("Age", min_value=20, max_value=80, value=50)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=90, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2])


def encode_onehot(value, classes):
    return [1 if value == cls else 0 for cls in classes]

numerical = ['age', 'trestbps', 'chol', 'thalach']

input_dict = {
    'age': age,
    'sex': sex,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'age_chol': age * chol,
    'age_range': pd.cut([age], bins=[20, 30, 40, 50, 60, 70, 80], labels=False)[0]
}

input_dict.update({
    **dict(zip(['cp_0', 'cp_1', 'cp_2', 'cp_3'], encode_onehot(cp, [0, 1, 2, 3]))),
    **dict(zip(['restecg_0', 'restecg_1', 'restecg_2'], encode_onehot(restecg, [0, 1, 2]))),
    **dict(zip(['slope_0', 'slope_1', 'slope_2'], encode_onehot(slope, [0, 1, 2]))),
    **dict(zip(['ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4'], encode_onehot(ca, [0, 1, 2, 3, 4]))),
    **dict(zip(['thal_0', 'thal_1', 'thal_2'], encode_onehot(thal, [0, 1, 2]))),
})

user_data = pd.DataFrame([input_dict])

user_data[numerical] = scaler.transform(user_data[numerical])

for col in features:
    if col not in user_data.columns:
        user_data[col] = 0

user_data = user_data[features]

if st.button("Predict"):
    prediction = model.predict(user_data)[0]
    chance = model.predict_proba(user_data)[0][1]

    if prediction == 1:
        st.error(f"High risk of heart disease with a {chance*100:.2f}% chance.")
    else:
        st.success(f"No heart disease predicted. Chance: {chance*100:.2f}%.")
