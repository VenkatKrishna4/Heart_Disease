import streamlit as st
import pickle
import numpy as np
import base64
import os

os.system("pip install -r requirements.txt")

# Loading the Models
stroke_model_path = "stroke_prediction_model.pkl"
chf_model_path = "chf_prediction_model.pkl"

try:
    with open(stroke_model_path, "rb") as file:
        stroke_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading Stroke Prediction model: {e}")

try:
    with open(chf_model_path, "rb") as file:
        chf_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading CHF Prediction model: {e}")

def set_background(image_url):
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

set_background("https://raw.githubusercontent.com/VenkatKrishna4/Heart_Disease/main/background.png")

st.title("🩺 Heart Disease Predictor")

disease_type = st.radio("Select the disease to predict:", ["Stroke", "Congestive Heart Failure (CHF)"])

# Stroke disesase
if disease_type == "Stroke":
    st.subheader("🧠 Stroke Prediction")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt job", "Never worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])

    # Convert inputs to numerical format
    gender = 1 if gender == "Male" else 0
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    ever_married = 1 if ever_married == "Yes" else 0
    work_type = ["Private", "Self-employed", "Govt job", "Children", "Never worked"].index(work_type)
    residence_type = 1 if residence_type == "Urban" else 0
    smoking_status = ["Never smoked", "Formerly smoked", "Smokes", "Unknown"].index(smoking_status)


    if st.button("Predict Stroke Risk"):
        features = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type,
                              residence_type, avg_glucose_level, bmi, smoking_status]])
        
        prediction = stroke_model.predict(features)

        if prediction[0] == 1:
            st.error("⚠️ High risk of stroke! Consult a doctor immediately.")
        else:
            st.success("✅ Low risk of stroke! Maintain a healthy lifestyle.")

# CHF Prediction
elif disease_type == "Congestive Heart Failure (CHF)":
    st.subheader("❤️ CHF Prediction")
    
    age = st.number_input("Age", min_value=1, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=500, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "Yes" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Up", "Flat", "Down"].index(st_slope)

    if st.button("Predict CHF Risk"):
        features = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr,
                              exercise_angina, oldpeak, st_slope]])
        
        prediction = chf_model.predict(features)

        if prediction[0] == 1:
            st.error("⚠️ High risk of Congestive Heart Failure! Consult a doctor immediately.")
        else:
            st.success("✅ Low risk of CHF! Maintain a healthy lifestyle.")
