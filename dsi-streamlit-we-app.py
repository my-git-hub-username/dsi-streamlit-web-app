import streamlit as st
import pandas as pd
import joblib

#load pipeline object
model = joblib.load("model.joblib")

#add title and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")

#to see what web app looks like so far:
    #anaconda prompt
    #enter dsi-streamlit virtual environment : conda activate dsi-streamlit-web-app
    #change directory to where code is
    #streamlit command to run app locally : streamlit run _file name_

#inputs: age, gender, credit score
age = st.number_input("01. Enter the customer's age", 18, 120, 35)
gender = st.radio("02. Select the customer's gender", ["M", "F"])
credit_score = st.number_input("03. Enter customer's credit score", 0,1000,500)

#submit inputs to model
if st.button("Submit for Prediction"):
    #store inputs as df
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    #apply pipeline and extract prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    #show prediction
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")
    