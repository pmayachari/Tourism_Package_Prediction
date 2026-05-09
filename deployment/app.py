import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="prabhusm/tourism_package_prediction_model", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a tourism package based on its parameters.
Please enter the data below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Enquiry"])
Occupation = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
MaritalStatus = st.selectbox("MaritalStatus", ["Divorced","Married","Single","Unmarried"])
Designation = st.selectbox("Designation", ["AVP","Executive","Manager","Senior Manager","VP"])
Passport = st.selectbox("Has Passport ?", ["No","Yes"])
OwnCar = st.selectbox("Owns Car ?", ["No","Yes"])
CityTier = st.selectbox("City Tier", ["1","2","3"])

Age = st.number_input("Age", min_value=1, max_value=200, value=25, step=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=0, max_value=90, value=5)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=100, value=1, step=1)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=0, max_value=300, value=3)
NumberOfTrips = st.number_input("Number Of Trips", min_value=0, max_value=100, value=1)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=100, value=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=1)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=25, value=1)
MonthlyIncome = st.number_input("Monthly Income", min_value=1, max_value=30000000, value=30000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    TypeofContact : TypeofContact,	
    Occupation  : Occupation,		
    Gender  : Gender,	
    ProductPitched  : ProductPitched,
    MaritalStatus : MaritalStatus,	
    Designation : Designation,
    Passport  : Passport,	
    OwnCar  : OwnCar,		
    CityTier  : CityTier,	
    Age : Age,		
    DurationOfPitch : DurationOfPitch,		
    NumberOfPersonVisiting  : NumberOfPersonVisiting,
    PreferredPropertyStar : PreferredPropertyStar,	
    NumberOfTrips : NumberOfTrips,		
    NumberOfFollowups : NumberOfFollowups,		
    PitchSatisfactionScore  : PitchSatisfactionScore,
    NumberOfChildrenVisiting :NumberOfChildrenVisiting,
    MonthlyIncome : MonthlyIncome
}])


if st.button("Predict Tourism Package"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Package" if prediction == 1 else "No Prediction"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
