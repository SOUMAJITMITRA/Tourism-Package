import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "tourism_project/deployment/best_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully.")
else:
    st.error(" Model file not found.")
    st.stop()

st.title("Wellness Tourism Package Purchase Prediction v3")
st.write("Fill in the customer details below:")

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
city_tier = st.selectbox("City Tier", [1, 2, 3])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=1)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0, value=50000)
pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)

# Categorical dropdowns (aligned with dataset)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

# Build input DataFrame with explicit casting
input_data = pd.DataFrame({
    "Age": [int(age)],
    "TypeofContact": [str(typeof_contact)],
    "CityTier": [int(city_tier)],
    "Occupation": [str(occupation)],
    "Gender": [str(gender)],
    "NumberOfPersonVisiting": [int(num_person_visiting)],
    "PreferredPropertyStar": [int(preferred_property_star)],
    "MaritalStatus": [str(marital_status)],
    "NumberOfTrips": [int(num_trips)],
    "Passport": [int(passport)],
    "OwnCar": [int(own_car)],
    "NumberOfChildrenVisiting": [int(num_children_visiting)],
    "Designation": [str(designation)],
    "MonthlyIncome": [int(monthly_income)],
    "PitchSatisfactionScore": [int(pitch_score)],
    "ProductPitched": [str(product_pitched)],
    "NumberOfFollowups": [int(num_followups)],
    "DurationOfPitch": [int(duration_pitch)]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Customer is likely to purchase (probability: {probability:.2f})")
    else:
        st.warning(f"Customer is unlikely to purchase (probability: {probability:.2f})")
