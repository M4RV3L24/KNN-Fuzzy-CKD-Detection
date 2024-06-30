import streamlit as st
import functionality as f
import pandas as pd
import numpy as np
import os
import joblib


st.header("Data")

if "new_data" in st.session_state:
    data = st.session_state['new_data']
else:
    st.switch_page("pages/2_Input.py")

st.table(data)

st.header("Result")
# Define the paths for the model and parameters
model_path = 'knn_best_model.pkl'
params_path = 'knn_best_params.pkl'


# Load and preprocess data
file_path = 'dataset/kidney_disease.csv'
features, target, preprocessor = f.load_and_preprocess_data(file_path)

# Split the data
x_train, x_test, y_train, y_test = f.train_test_split(features, target, test_size=0.2, stratify=target)

# Train the KNN model
knn_best, best_params = f.train_knn_model(x_train, y_train, preprocessor)


# # Evaluate the model
y_pred = knn_best.predict(x_test)
accuracy = f.accuracy_score(y_pred, y_test)

st.write(f'Best parameters: {best_params}')
st.write(f'Accuracy: {accuracy}')


data['gfr'] = data.apply(f.calculate_gfr, axis=1)
data['bun'] = data.apply(f.calculate_bun, axis=1)
st.write(data["gfr"])
st.write(data["bun"])


y_pred = knn_best.predict(data)
y_pred_probabilities = knn_best.predict_proba(data)



st.write(y_pred)

if y_pred[0] == 0:
    st.write("The patient is diagnosed with Chronic Kidney Disease.")
else:
    st.write("The patient is not diagnosed with Chronic Kidney Disease.")


