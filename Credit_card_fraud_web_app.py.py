# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import pickle 
import streamlit as st

#loading the saved model 

loaded_model = pickle.load(open('C:/Users/ABC/OneDrive/Desktop/creditcard.pkl', 'rb'))

# Function to simulate the prediction process
def predict_fraud(time, amount, features):
    input_data = [time] + features + [amount]
    input_df = pd.DataFrame([input_data], columns=[f'Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
    prediction = loaded_model.predict(input_df)
    return 'Fraudulent transaction' if prediction[0] == 1 else 'Legitimate transaction'

# Define the Streamlit app
def main():
    # Giving title
    st.title('Credit Card Fraud Detection')
    
    st.write("""
    ### Enter the transaction details:
    """)
    
    # Inputs for the model
    time = st.number_input('Time', min_value=0.0, step=0.1)
    amount = st.number_input('Amount', min_value=0.0, step=0.1)
    features = [st.number_input(f'V{i}', value=0.0, step=0.1) for i in range(1, 29)]

    if st.button('Predict'):
        result = predict_fraud(time, amount, features)
        st.write(f'Prediction: {result}')

if __name__ == '__main__':
    main()
    