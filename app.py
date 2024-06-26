import streamlit as st
import pandas as pd
import joblib

# Load your saved model
rf_model = joblib.load('churn_predict_model')  

# Load dataset to get statistical values for input sliders
df = pd.read_csv('Churn_Modelling.csv')

# Input Parameters
st.header('Customer Churn Prediction')
st.sidebar.header('User Input Parameters')

def user_input_features():
    credit_score = st.sidebar.number_input('Credit Score', min_value=int(df['CreditScore'].min()), max_value=int(df['CreditScore'].max()), value=int(df['CreditScore'].mean()), key='credit_score')
    geography = st.sidebar.selectbox('Geography', df['Geography'].unique(), key='geography')
    gender = st.sidebar.selectbox('Gender', df['Gender'].unique(), key='gender')
    age = st.sidebar.number_input('Age', min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].mean()), key='age')
    tenure = st.sidebar.number_input('Tenure', min_value=int(df['Tenure'].min()), max_value=int(df['Tenure'].max()), value=int(df['Tenure'].mean()), key='tenure')
    balance = st.sidebar.number_input('Balance', min_value=float(df['Balance'].min()), max_value=float(df['Balance'].max()), value=float(df['Balance'].mean()), key='balance')
    num_of_products = st.sidebar.number_input('Number of Products', min_value=int(df['NumOfProducts'].min()), max_value=int(df['NumOfProducts'].max()), value=int(df['NumOfProducts'].mean()), key='num_of_products')
    has_credit_card = st.sidebar.selectbox('Has Credit Card', [0, 1], key='has_credit_card')
    is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1], key='is_active_member')
    estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=float(df['EstimatedSalary'].min()), max_value=float(df['EstimatedSalary'].max()), value=float(df['EstimatedSalary'].mean()), key='estimated_salary')
    
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_credit_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Show the input parameters
st.table(input_df.transpose())


# Preprocess the input data
def preprocess_input_data(df):
    df["Geography"] = df["Geography"].apply(lambda x: 1 if x == "France" else (2 if x == "Spain" else 3))
    df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "Female" else 2)
    return df

if st.button('Predict'):
    processed_df = preprocess_input_data(input_df)

    # Predict using the loaded model
    prediction = rf_model.predict(processed_df)

    # Display prediction
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('Churn Status: Churned 😢')
    else:
        st.write('Churn Status: Not Churned')
        st.balloons()
