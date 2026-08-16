import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "tourism_project/deployment/best_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully.")
else:
    st.error("Model file not found.")
    st.stop()

st.title("Wellness Tourism Package Purchase Prediction v2")
st.write("Fill in the customer details below:")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=1)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.text_input("Designation", "Manager")
monthly_income = st.number_input("Monthly Income", min_value=0, value=50000)
pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
product_pitched = st.text_input("Product Pitched", "Basic")
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)

input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [typeof_contact],
    "CityTier": [city_tier],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [num_person_visiting],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [num_trips],
    "Passport": [passport],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [num_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income],
    "PitchSatisfactionScore": [pitch_score],
    "ProductPitched": [product_pitched],
    "NumberOfFollowups": [num_followups],
    "DurationOfPitch": [duration_pitch]
})


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 Customer is likely to purchase (probability: {probability:.2f})")
    else:
        st.warning(f"🙁 Customer is unlikely to purchase (probability: {probability:.2f})")
