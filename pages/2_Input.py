import streamlit as st
import functionality
import pandas as pd
import numpy as np

st.header("Patient's Data")

patient_form = st.form('new_input')

with patient_form:
    age = st.number_input("Age", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
    sg = st.number_input("Specific Gravity", min_value=0.00, step=0.005, format="%.4f")
    al = st.selectbox("Albumin", ['-',0,1,2,3,4,5])
    su = st.selectbox("Sugar", ['-',0,1,2,3,4,5])
    bgr= st.number_input("Blood Glucose Random", min_value=0.0, step=0.1)
    bu = st.number_input("Blood Urea", min_value=0.0, step=0.1)
    sc = st.number_input("Serum Creatinine", min_value=0.0, step=0.1)
    sod = st.number_input("Sodium", min_value=0.0, step=0.1)
    pot = st.number_input("Potassium", min_value=0.0, step=0.1)
    hemo = st.number_input("Hemoglobin", min_value=0.0, step=0.1)
    pcv = st.number_input("Packed Cell Volume", min_value=0.0, step=0.1)
    wc = st.number_input("White Blood Cell Count", min_value=0.0, step=0.1)
    rc = st.number_input("Red Blood Cell Count", min_value=0.0, step=0.1)
    rbc = st.selectbox("Red Blood Cells", ['-', 'normal', 'abnormal'])
    pc = st.selectbox("Pus Cell", ['-', 'normal', 'abnormal'])
    pcc = st.selectbox("Pus Cell Clumps", ['-','notpresent', 'present'])
    ba = st.selectbox("Bacteria", ['-','notpresent', 'present'])
    htn = st.selectbox("Hypertension", ['-','yes', 'no'])
    dm = st.selectbox("Diabetes Mellitus", ['-','yes', 'no'])
    cad = st.selectbox("Coronary Artery Disease", ['-','yes', 'no'])
    appet = st.selectbox("Appetite", ["-",'good', 'poor'])
    pe = st.selectbox("Pedal Edema", ["-",'yes', 'no'])
    ane = st.selectbox("Anemia", ["-",'yes', 'no'])
    submit_button = st.form_submit_button('Submit')



if submit_button:
    
    frame = {
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'su': [su],
        'bgr': [bgr],
        'bu': [bu],
        'sc': [sc],
        'sod': [sod],
        'pot': [pot],
        'hemo': [hemo],
        'pcv': [pcv],
        'wc': [wc],
        'rc': [rc],
        'rbc': [rbc],
        'pc': [pc],
        'pcc': [pcc],
        'ba': [ba],
        'htn': [htn],
        'dm': [dm],
        'cad': [cad],
        'appet': [appet],
        'pe': [pe],
        'ane': [ane], 
    }

    for key in frame:
        if key != 'al' and key != 'su':
            if frame[key][0] == 0:
                frame[key] = [np.nan]



    new_data = pd.DataFrame(frame)

    
    try:
        history = pd.read_csv('history.csv')
        history = pd.concat([history, new_data], ignore_index=True)
    except FileNotFoundError:
        history = new_data

    
    history.to_csv('history.csv', index=False)

    
    st.session_state["new_data"] = pd.DataFrame(frame)

   
    st.switch_page("pages/3_Check Result.py")


