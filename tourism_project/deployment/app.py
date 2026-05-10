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
Age = st.number_input("Age", min_value=18, max_value=100, value=30)

CityTier = st.selectbox("City Tier",[1, 2, 3])

DurationOfPitch = st.number_input("Duration Of Pitch",min_value=1,max_value=30,value=10)

NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting",min_value=1,max_value=10,value=2)

NumberOfFollowups = st.number_input("Number Of Followups",min_value=0,max_value=10,value=2)

PreferredPropertyStar = st.selectbox("Preferred Property Star",[1, 2, 3, 4, 5])

NumberOfTrips = st.number_input("Number Of Trips", min_value=0, max_value=50,value=2)

Passport = st.selectbox("Passport",["No", "Yes"])

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score",1,5,3)

OwnCar = st.selectbox("Own Car", ["No", "Yes"])

NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting",min_value=0,max_value=10,value=0)

MonthlyIncome = st.number_input("Monthly Income",min_value=1000,max_value=1000000,value=50000)


# Dropdown Mapping
TypeofContact_map = {
    "Self Enquiry": 0,
    "Company Invited": 1
}

Occupation_map = {
    "Salaried": 0,
    "Small Business": 1,
    "Large Business": 2,
    "Free Lancer": 3
}

Gender_map = {
    "Male": 0,
    "Female": 1
}

ProductPitched_map = {
    "Basic": 0,
    "Standard": 1,
    "Deluxe": 2,
    "Super Deluxe": 3,
    "King": 4
}

MaritalStatus_map = {
    "Single": 0,
    "Married": 1,
    "Divorced": 2
}

Designation_map = {
    "Executive": 0,
    "Manager": 1,
    "Senior Manager": 2,
    "VP": 3,
    "AVP": 4
}


TypeofContact = st.selectbox("Type Of Contact",list(TypeofContact_map.keys()))

Occupation = st.selectbox("Occupation",list(Occupation_map.keys()))

Gender = st.selectbox("Gender", list(Gender_map.keys()))

ProductPitched = st.selectbox("Product Pitched",list(ProductPitched_map.keys()))

MaritalStatus = st.selectbox("Marital Status",list(MaritalStatus_map.keys()))

Designation = st.selectbox("Designation",list(Designation_map.keys()))


if st.button("Predict"):

    input_data = pd.DataFrame([{
        'Age': Age,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'PreferredPropertyStar': PreferredPropertyStar,
        'NumberOfTrips': NumberOfTrips,
        'Passport': 1 if Passport == "Yes" else 0,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': 1 if OwnCar == "Yes" else 0,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'MonthlyIncome': MonthlyIncome,

        # Converted categorical values
        'TypeofContact': TypeofContact_map[TypeofContact],
        'Occupation': Occupation_map[Occupation],
        'Gender': Gender_map[Gender],
        'ProductPitched': ProductPitched_map[ProductPitched],
        'MaritalStatus': MaritalStatus_map[MaritalStatus],
        'Designation': Designation_map[Designation]
    }])

    # Ensure string column names
    input_data.columns = input_data.columns.astype(str)

    st.write("Input Data")
    st.dataframe(input_data)

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Customer is likely to purchase tourism package")
    else:
        st.error("Customer is unlikely to purchase tourism package")
