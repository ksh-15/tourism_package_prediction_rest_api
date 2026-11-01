import os
import joblib
import pandas as pd
import streamlit as st

# Configure page style
st.set_page_config(page_title="Wellness Package Prediction", layout="centered")
st.title("🌴 Visit with Us: Wellness Package Predictor")
st.write("Enter customer parameters below to evaluate the likelihood of package purchase.")

# Load the pre-trained pipeline model
# In local/Git layout, we check both relative paths
model_paths = [
    "tourism_project/model_building/model.joblib",
    "../model_building/model.joblib",
    "model.joblib"
]

model = None
for path in model_paths:
    if os.path.exists(path):
        model = joblib.load(path)
        break

if model is None:
    st.error("⚠️ Trained model artifact could not be found! Please ensure 'model.joblib' is compiled and saved.")
else:
    st.success("✅ Predictive Model Loaded Successfully!")

    # Setup forms and inputs
    st.subheader("Customer Profiling")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        TypeofContact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
        CityTier = st.selectbox("City Tier", [1, 2, 3], index=0)
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
        Gender = st.selectbox("Gender", ["Female", "Male"])
        NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3, step=1)
        PreferredPropertyStar = st.slider("Preferred Property Rating", 3.0, 5.0, 4.0, 0.5)

    with col2:
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        NumberOfTrips = st.number_input("Average Annual Trips", min_value=1, max_value=25, value=3, step=1)
        Passport = st.selectbox("Has Valid Passport?", ["Yes", "No"])
        OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])
        NumberOfChildrenVisiting = st.number_input("Number of Children Accompanying", min_value=0, max_value=10, value=1, step=1)
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Monthly Gross Income ($)", min_value=1000.0, max_value=150000.0, value=25000.0, step=100.0)

    st.subheader("Sales Rep Interaction")
    col3, col4 = st.columns(2)
    with col3:
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3, 1)
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    with col4:
        NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=15, value=3, step=1)
        DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=300.0, value=15.0, step=1.0)

    # Encode indicators
    passport_val = 1 if Passport == "Yes" else 0
    own_car_val = 1 if OwnCar == "Yes" else 0

    # Save raw inputs inside a pandas dataframe matching pipeline layout
    input_dict = {
        'Age': [float(Age)],
        'TypeofContact': [TypeofContact],
        'CityTier': [int(CityTier)],
        'DurationOfPitch': [float(DurationOfPitch)],
        'Occupation': [Occupation],
        'Gender': [Gender],
        'NumberOfPersonVisiting': [int(NumberOfPersonVisiting)],
        'NumberOfFollowups': [float(NumberOfFollowups)],
        'ProductPitched': [ProductPitched],
        'PreferredPropertyStar': [float(PreferredPropertyStar)],
        'MaritalStatus': [MaritalStatus],
        'NumberOfTrips': [float(NumberOfTrips)],
        'Passport': [int(passport_val)],
        'PitchSatisfactionScore': [int(PitchSatisfactionScore)],
        'OwnCar': [int(own_car_val)],
        'NumberOfChildrenVisiting': [float(NumberOfChildrenVisiting)],
        'Designation': [Designation],
        'MonthlyIncome': [float(MonthlyIncome)]
    }

    input_df = pd.DataFrame(input_dict)

    # Trigger Prediction
    st.write("--- ")
    if st.button("🎯 Predict Purchase Likelihood", use_container_width=True):
        # Run prediction
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Results")
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            if pred == 1:
                st.metric(label="Recommendation", value="🎯 Target Customer")
            else:
                st.metric(label="Recommendation", value="❌ Pass (Unlikely)")

        with col_res2:
            st.metric(label="Purchase Confidence Score", value=f"{proba*100:.2f}%")

        if pred == 1:
            st.balloons()
            st.info("Strong Candidate! High probability of booking the Wellness Tourism Package.")
        else:
            st.warning("Low Candidate! Focus sales efforts on prospects with stronger indicators.")
