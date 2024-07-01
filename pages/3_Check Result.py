import streamlit as st
import functionality as f
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
# st.header("Patient's Diagnosis")
st.subheader("+ Patient's Data")

age = st.sidebar.expander("Age (age)")
age.write("The age of the patient in years.")
bp = st.sidebar.expander("Blood Pressure (bp)")
bp.write("The patient's blood pressure, typically measured in mmHg.")
sg = st.sidebar.expander("Specific Gravity (sg)")
sg.write("A measure of the concentration of solutes in the urine.")
al = st.sidebar.expander("Albumin (al)")
al.write("The presence of albumin in the urine, an indicator of kidney function.")
su = st.sidebar.expander("Sugar (su)")
su.write("The presence of sugar in the urine, an indicator of diabetes.")
rbc = st.sidebar.expander("Red Blood Cells (rbc)")
rbc.write("The count of red blood cells in the urine.")
pc = st.sidebar.expander("Pus Cell (pc)")
pc.write("The count of pus cells in the urine, indicating infection.")
pcc = st.sidebar.expander("Pus Cell Clumps (pcc)")
pcc.write("The presence of clumps of pus cells in the urine.")
ba = st.sidebar.expander("Bacteria (ba)")
ba.write("The presence of bacteria in the urine.")
bgr = st.sidebar.expander("Blood Glucose Random (bgr)")
bgr.write("Random blood glucose level.")
bu = st.sidebar.expander("Blood Urea (bu)")
bu.write("The level of urea in the blood, an indicator of kidney function.")
sc = st.sidebar.expander("Serum Creatinine (sc)")
sc.write("The level of creatinine in the blood, an indicator of kidney function.")
sod = st.sidebar.expander("Sodium (sod)")
sod.write("The level of sodium in the blood.")
pot = st.sidebar.expander("Potassium (pot)")
pot.write("The level of potassium in the blood.")
hemo = st.sidebar.expander("Hemoglobin (hemo)")
hemo.write("The level of hemoglobin in the blood.")
pcv = st.sidebar.expander("Packed Cell Volume (pcv)")
pcv.write("The volume percentage of red blood cells in blood.")
wc = st.sidebar.expander("White Blood Cell Count (wc)")
wc.write("The count of white blood cells in the blood.")
rc = st.sidebar.expander("Red Blood Cell Count (rc)")
rc.write("The count of red blood cells in the blood.")
htn = st.sidebar.expander("Hypertension (htn)")
htn.write("Indicates if the patient has hypertension.")
dm = st.sidebar.expander("Diabetes Mellitus (dm)")
dm.write("Indicates if the patient has diabetes.")
cad = st.sidebar.expander("Coronary Artery Disease (cad)")
cad.write("Indicates if the patient has coronary artery disease.")
appet = st.sidebar.expander("Appetite (appet)")
appet.write("The patient's appetite.")
pe = st.sidebar.expander("Pedal Edema (pe)")
pe.write("Indicates if the patient has pedal edema.")
ane = st.sidebar.expander("Anemia (ane)")
ane.write("Indicates if the patient has anemia.")
classification = st.sidebar.expander("Classification (classification)")
classification.write("The classification of the patient's condition.")


if "new_data" in st.session_state:
    data = st.session_state['new_data']
else:
    st.switch_page("pages/2_Input.py")

st.table(data)

st.subheader("+ Prediction")
# Define the paths for the model and parameters
model_path = 'knn_best_model.pkl'
params_path = 'knn_best_params.pkl'


# Load and preprocess data
file_path = 'dataset/kidney_disease.csv'
features, target, preprocessor = f.load_and_preprocess_data(file_path)

# Split the data
x_train, x_test, y_train, y_test = f.train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

# Train the KNN model
if os.path.exists(model_path) and os.path.exists(params_path):
    knn_best= joblib.load("knn_best_model.pkl")
    best_params= joblib.load("knn_best_params.pkl")

else:
    knn_best, best_params = f.train_knn_model(x_train, y_train, preprocessor)


# # Evaluate the model
y_pred = knn_best.predict(x_test)
accuracy = f.accuracy_score(y_pred, y_test)


col1, col2 = st.columns([2, 1])
metric, neighbours, weight = col1.columns(3)
with metric:
    st.write('Metric: ', best_params["classifier__metric"])

with neighbours:
    st.write('Neighbours: ', best_params['classifier__n_neighbors'])

with weight:
    st.write('Weight: ', best_params['classifier__weights'])

col2.write(f'Accuracy: {accuracy}')

data['gfr'] = data.apply(f.calculate_gfr, axis=1)
data['bun'] = data.apply(f.calculate_bun, axis=1)

st.write(data)

y_pred = knn_best.predict(data)
y_pred_probabilities = knn_best.predict_proba(data)

if y_pred[0] == 0:
    st.text("The patient is diagnosed with Chronic Kidney Disease.")
else:
    st.text("The patient is not diagnosed with Chronic Kidney Disease.")



#fuzzy

fuzzy_universe = f.fuzzy_init()
gfr_member = f.gfr_member(fuzzy_universe["gfr"])
bun_member = f.bun_member(fuzzy_universe["bun"])
creatinine_member = f.creatinine_member(fuzzy_universe["creatinine"])
hemoglobin_member = f.hemoglobin_member(fuzzy_universe["hemoglobin"])
bp_member = f.bp_member(fuzzy_universe["bp"])
albuminuria_member = f.albuminuria_member(fuzzy_universe["albuminuria"])
sodium_member = f.sodium_member(fuzzy_universe["sodium"])
potassium_member = f.potassium_member(fuzzy_universe["potassium"])
severity_member = f.severity_member(fuzzy_universe["severity"])

simulation = f.create_fuzzy_rules(gfr_member, creatinine_member, bun_member, albuminuria_member, bp_member, 
                                  hemoglobin_member, sodium_member, potassium_member, severity_member)



simulation.input['gfr'] = data["gfr"]
simulation.input['creatinine'] = data["sc"]
simulation.input['bun'] = data["bun"]
simulation.input['albuminuria'] = data["al"]  
simulation.input['bp'] = data["bp"]  
simulation.input['hemoglobin'] = data["hemo"]  
simulation.input['sodium'] = data["sod"]  
simulation.input['potassium'] = data["pot"]  

# Compute the result
simulation.compute()

# Print the severity level
st.write(f"Severity level: {simulation.output['severity']:.2f}")

# Visualize the severity graph

# # Save the severity graph to a file
# plt.savefig('severity_graph.png')
# print("Severity graph saved as 'severity_graph.png'")

# image = Image.open('severity_graph.png')
# image.show()

# Visualize the final aggregated result
fig, ax0 = plt.subplots(figsize=(8, 3))

severity_member.view(sim=simulation, ax=ax0)
st.pyplot(fig)






