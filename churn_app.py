import streamlit as st
import pandas as pd
import joblib

# Load model + scaler
model = joblib.load('Best_model_churn.pkl')
scaler = joblib.load('scaler_churn.pkl')

st.title("  Customer Churn Prediction App")

st.markdown("###   Customer Information")

 # INPUTS


credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 42)
tenure = st.number_input("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 60000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

st.markdown("###   Geography")
geo_france = st.selectbox("France", [0, 1])
geo_spain = st.selectbox("Spain", [0, 1])
geo_germany = st.selectbox("Germany", [0, 1])

st.markdown("###   Gender")
gender_male = st.selectbox("Male (1 = Male, 0 = Female)", [0, 1])
 
# PREDICTION
 
if st.button("🔍 Predict Churn"):

    data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary,
        'Geography_France': geo_france,
        'Geography_Spain': geo_spain,
        'Geography_Germany': geo_germany,
        'Gender_Male': gender_male
    }])

    # scaling
    data_scaled = scaler.transform(data)

    # prediction
    pred = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0][1]

    # result
    st.subheader("  Result")

    if pred == 1:
        st.error("  Client will CHURN (leave the bank)")
    else:
        st.success("  Client will STAY")

    st.metric("Churn Probability", f"{proba:.2%}")
    st.progress(float(proba))


    #  streamlit run churn_app.py