import streamlit as st
import pandas as pd
import joblib
import os

# Path to the trained model file
MODEL_PATH = "tourism_project/deployment/best_model.pkl"

# Load the trained model if it exists, otherwise stop execution
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully.")
else:
    st.error("Model file not found.")
    st.stop()

# Application title and description
st.title("Wellness Tourism Package Purchase Prediction")
st.write("Fill in the customer details below:")

# -----------------------------
# Collect numeric inputs from the user
# -----------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
city_tier = st.selectbox("City Tier", [1, 2, 3])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=1)
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0, value=50000)
pitch_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)

# -----------------------------
# Collect binary inputs (Yes/No mapped to 1/0)
# -----------------------------
passport_option = st.selectbox("Passport", ["No", "Yes"])
passport = 1 if passport_option == "Yes" else 0

own_car_option = st.selectbox("Own Car", ["No", "Yes"])
own_car = 1 if own_car_option == "Yes" else 0

# -----------------------------
# Collect categorical inputs (dropdowns aligned with dataset categories)
# -----------------------------
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

# -----------------------------
# Build input DataFrame for prediction
# Explicit casting ensures correct data types
# -----------------------------
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

# -----------------------------
# Prediction logic
# -----------------------------
if st.button("Predict"):
    # Generate prediction and probability from the trained model
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display result based on prediction
    if prediction == 1:
        st.success(f"Customer is likely to purchase (probability: {probability:.2f})")
    else:
        st.warning(f"Customer is unlikely to purchase (probability: {probability:.2f})")
