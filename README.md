# Customer Churn Prediction App

This Streamlit application predicts customer churn using a machine learning model trained on data from a bank's customer dataset ('Churn_Modelling.csv'). The model used in this application is a Random Forest classifier.

## Application Overview

The application allows users to input various parameters related to a customer and predicts whether the customer is likely to churn based on those inputs. It provides a user-friendly interface where users can interact with sliders and dropdowns to input data.

![Screenshot 2024-06-26 154210](https://github.com/Tanzila-Ikhlaq/CustomerChurnPrediction/assets/141930681/cd2e8f03-2a11-43fc-8f04-253ceeb5170e)

## Features

- **Input Parameters**: Users can input customer details such as credit score, geography, gender, age, tenure, balance, number of products, presence of a credit card, activity status, and estimated salary.
- **Prediction**: Upon clicking the 'Predict' button, the application preprocesses the input data and uses a pre-trained Random Forest model to predict the churn status.
- **Output**: The application displays whether the predicted churn status is "Churned 😢" or "Not Churned". If the prediction is "Not Churned", it celebrates with balloons.

## Technologies Used

- Python
- Streamlit
- Pandas
- Joblib (for model persistence)

## Files Included

- `app.py`: Main script containing the Streamlit application code.
- `Churn_Modelling.csv`: Dataset used for training the model.
- `churn_predict_model`: Serialized Random Forest model.
- `customer_churn_prediction.ipynb`: Jupyter Notebook containing EDA and model training.
  

