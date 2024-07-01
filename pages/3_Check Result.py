import streamlit as st
import functionality as f
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt


# Menambahkan CSS untuk FontAwesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    """, unsafe_allow_html=True)


# st.header("Patient's Diagnosis")
st.markdown("""
    <h2 style='text-align: center; '>👤Patient's Data </h1>
    """, unsafe_allow_html=True)


if "new_data" in st.session_state:
    data = st.session_state['new_data']
else:
    st.switch_page("pages/2_Input.py")

st.write(data)

st.markdown("""
    <h2 style='text-align: center; '>📊 Prediction </h1>
    """, unsafe_allow_html=True)

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


y_pred = knn_best.predict(data)
y_pred_probabilities = knn_best.predict_proba(data)

if y_pred[0] == 0:
    st.text("The patient is diagnosed with Chronic Kidney Disease.")
else:
    st.text("The patient is not diagnosed with Chronic Kidney Disease.")


def print_chart(name,key,min,max,step,list_options=["low", "normal", "high"]):
    col1,col2 = st.columns([3,2])
    index_values =  st.session_state.get(key, {})

    with col1:
        expander = st.expander(f"{name} Advanced Settings")
        # option
        option_key = f"{name}_option"
        option = expander.selectbox(
            "Which one to change",
            (list_options), key=option_key)
        
        
        # print berdasarkan option dan jumlah index_values
        if option in index_values:
            if len(index_values[option]) == 3:
                a1, a2, a3 = index_values[option]
                a4 = None
            elif len(index_values[option]) == 4:
                a1, a2, a3, a4 = index_values[option]
        else:
            a1, a2, a3, a4 = 0.0, 0.0, 0.0, 0.0


        new_a1 = expander.slider(f"{name}1", min_value=min, max_value=max, value=a1, step=step)
        new_a2 = expander.slider(f"{name}2", min_value=min, max_value=max, value=a2, step=step)
        new_a3 = expander.slider(f"{name}3", min_value=min, max_value=max, value=a3, step=step)
        if a4 is not None:
            new_a4 = expander.slider(f"{name}4", min_value=min, max_value=max, value=a4, step=step)
        else:
            new_a4 = None

        # Validation logic
        if new_a2 < new_a1:
            st.warning(f"Value '{name}2' must be greater than '{name}1'. Adjusting values...")
            new_a2 = new_a1

        if new_a3 < new_a2:
            st.warning(f"Value '{name}3' must be greater than '{name}2'. Adjusting values...")
            new_a3 = new_a2

        if new_a4 is not None and new_a4 < new_a3:
            st.warning(f"Value '{name}4' must be greater than '{name}3'. Adjusting values...")
            new_a4 = new_a3

        if new_a4 is not None:
            st.session_state[f'{name}_values'][option] = [new_a1, new_a2, new_a3, new_a4]
        else:
            st.session_state[f'{name}_values'][option] = [new_a1, new_a2, new_a3]
        
    with col2: 
        function_name = f"{name}_member"
        try:
            # Get the function reference from module f
            member_function = getattr(f, function_name)
            
            # Call the function and return its result
            result = member_function(fuzzy_universe[name], index_values)
            return result
        except AttributeError:
            st.error(f"Function {function_name} does not exist in module f.")
            return None
        

st.write(data[['gfr', 'sc', 'bun', 'al', 'bp', 'hemo', 'sod', 'pot']])
        
#fuzzy

fuzzy_universe = f.fuzzy_init()
# gfr_member = f.gfr_member(fuzzy_universe["gfr"])
# bun_member = f.bun_member(fuzzy_universe["bun"])
# creatinine_member = f.creatinine_member(fuzzy_universe["creatinine"])
# hemoglobin_member = f.hemoglobin_member(fuzzy_universe["hemoglobin"])
# bp_member = f.bp_member(fuzzy_universe["bp"])
# albuminuria_member = f.albuminuria_member(fuzzy_universe["albuminuria"])
# sodium_member = f.sodium_member(fuzzy_universe["sodium"])
# potassium_member = f.potassium_member(fuzzy_universe["potassium"])

gfr_member = print_chart(name="gfr",key="gfr_values",min=0.0,max=200.0,step=1.0,list_options=["stage 1", "stage 2", "stage 3", "stage 4", "stage 5"])
bun_member = print_chart(name="bun",key="bun_values",min=0.0,max=100.0,step=1.0,list_options=["very low", "low", "medium", "high", "very high"])
creatinine_member = print_chart(name="creatinine",key="creatinine_values",min=0.0,max=10.0,step=0.1,list_options=["very low", "low", "medium", "high", "very high"])
hemoglobin_member = print_chart(name="hemoglobin",key="hemoglobin_values",min=5.0,max=20.0,step=0.1)
bp_member = print_chart(name="bp",key="bp_values",min=50.0,max=200.0,step=1.0,list_options=["normal", "high", "very high"])
albuminuria_member = print_chart(name="albuminuria",key="albuminuria_values",min=0.0,max=6.0,step=1.0,list_options=["normal", "trace", "low", "medium", "high", "very high"])
sodium_member = print_chart(name="sodium",key="sodium_values",min=120.0,max=150.0,step=0.1)
potassium_member = print_chart(name="potassium",key="potassium_values",min=2.0,max=7.0,step=0.1)
def severity_chart():
    col1,col2 = st.columns([3,2])
    severity_values = st.session_state['severity_values']

    with col1:
        expander = st.expander("Severe Advanced Settings")
        # option
        option = expander.selectbox(
            "Which one to change",
            ("low", "medium", "high"))
        
        if option in severity_values:
            a1, a2, a3 = severity_values[option]
        else:
            a1, a2, a3 = 0.0, 0.0, 0.0  # Default values if not initialized

        new_a1 = expander.slider("severe1", min_value=0.0, max_value=1.1, value=a1, step=0.1)
        new_a2 = expander.slider("severe2", min_value=0.0, max_value=1.1, value=a2, step=0.1)
        new_a3 = expander.slider("severe3", min_value=0.0, max_value=1.1, value=a3, step=0.1)

        # Validation logic
        if new_a2 < new_a1:
            st.warning("Value 'severe2' must be greater than 'severe1'. Adjusting values...")
            new_a2 = new_a1  # Adjusting a2 to be greater than a1

        if new_a3 < new_a2:
            st.warning("Value 'severe3' must be greater than 'severe2'. Adjusting values...")
            new_a3 = new_a2   # Adjusting a3 to be greater than a2
        
        st.session_state['severity_values'][option] = [new_a1, new_a2, new_a3]
    with col2:
        severity_member = f.severity_member(fuzzy_universe["severity"], st.session_state['severity_values'])
    return severity_member

severity_member = severity_chart()
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

# Visualize the severity graph

severity_member.view(sim=simulation)

# # Save the severity graph to a file
plt.savefig('severity_graph.png')
print("Severity graph saved as 'severity_graph.png'")

@st.experimental_dialog("Patient Severity Level")
def showResult():
    st.image('severity_graph.png', caption='Kidney Severity')
    st.text(f"severity level: {simulation.output['severity']:.2f}")


# if st.button("show plot"):
#     showResult()

if st.sidebar.button("Severity Result", use_container_width=True, type="primary"):
    showResult()



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


    





