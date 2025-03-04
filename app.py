import streamlit as st
import pickle
import numpy as np
import base64

# Load the trained model
model_path = "stroke_prediction_model.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit UI
st.title("🧠 Stroke Prediction System")
st.write("Enter patient details to predict the risk of stroke.")

def set_background(image_file):
    with open(image_file, "rb") as img:
        base64_img = base64.b64encode(img.read()).decode()
    
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{base64_img}");
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

# Set local background image
set_background("C:/Users/gudur/Medibot_project/background.png")

# Input fields
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

# Prediction button
if st.button("Predict Stroke Risk"):
    features = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type,
                          residence_type, avg_glucose_level, bmi, smoking_status]])
    
    prediction = model.predict(features)
    

    if prediction[0] == 1:
        st.error("⚠️ High risk of stroke! Consult a doctor immediately.")
    else:
        st.success("✅ Low risk of stroke! Maintain a healthy lifestyle.")
