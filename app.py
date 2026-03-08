import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="CVD Risk Predictor", layout="wide")

st.markdown("# ❤️ Heart Disease & Stroke Risk Predictor")
st.write("Predicts risk for heart disease, stroke, or both using a Ridge Regression model.")

# Load the combined model you already have
@st.cache_resource
def load_model():
    return joblib.load('final_ridge_cvd_model.pkl')

model = load_model()
st.success("Model loaded successfully!")

# Sidebar Inputs
with st.sidebar:
    st.header("Patient Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 10, 100, 50)
    hypertension = st.selectbox("Hypertension (1=Yes, 0=No)", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.slider("Avg Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Prediction
if st.button("Predict Risk"):
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status],
        'age_group': ['middle_age'],
        'glucose_group': ['normal'],
        'bmi_group': ['normal']
    })

    # Compute engineered features manually
    if age <= 18: input_data['age_group'] = 'child'
    elif age <= 40: input_data['age_group'] = 'young_adult'
    elif age <= 60: input_data['age_group'] = 'middle_age'
    else: input_data['age_group'] = 'senior'

    if avg_glucose_level <= 100: input_data['glucose_group'] = 'normal'
    elif avg_glucose_level <= 126: input_data['glucose_group'] = 'prediabetes'
    elif avg_glucose_level <= 200: input_data['glucose_group'] = 'diabetes'
    else: input_data['glucose_group'] = 'high'

    if bmi < 18.5: input_data['bmi_group'] = 'underweight'
    elif bmi < 25: input_data['bmi_group'] = 'normal'
    elif bmi < 30: input_data['bmi_group'] = 'overweight'
    else: input_data['bmi_group'] = 'obese'

    for col in input_data.columns:
        if input_data[col].dtype == 'object':
            input_data[col] = input_data[col].astype(str)

    # Predict
    prediction = model.predict(input_data)[0]
    score = model.decision_function(input_data)[0]

    # 4-way approximation using score magnitude
    if prediction == 0:
        st.markdown("### ✅ **Low risk** — neither detected")
        st.success("No strong evidence of heart disease or stroke.")
    else:
        if score > 2.5:
            st.markdown("### ⚠️⚠️ **Both** heart disease and stroke likely")
            st.error("Very strong risk signal — high chance of both conditions.")
        elif score > 1.0:
            st.markdown("### ❤️ **Heart disease dominant** (possible stroke too)")
            st.warning("Stronger signal for heart disease; stroke risk may be present.")
        else:
            st.markdown("### 🧠 **Stroke dominant** (possible heart disease too)")
            st.warning("Stronger signal for stroke; heart disease risk may be present.")

    st.write(f"Decision score: {score:.3f}")
    st.subheader("Patient Summary")
    st.table(input_data.T.rename(columns={0: "Value"}))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 30px 0; font-size: 0.95em;'>
        Built in <strong>BOUESTI</strong> • Benin City, Edo State • March 2026<br>
        Educational project • Not for clinical use • Always consult a doctor
    </div>
    """,
    unsafe_allow_html=True
)