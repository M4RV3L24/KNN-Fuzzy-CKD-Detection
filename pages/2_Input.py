import streamlit as st
import functionality
import pandas as pd
import numpy as np

st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    """, unsafe_allow_html=True)


st.markdown("""
    <h1 style='text-align: center; color: black;'>👤Patient's Data </h1>
    """, unsafe_allow_html=True)
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


        
    # List of required keys
    # required_keys = ['age', 'sc', 'bu', 'al', 'bp', 'hemo', 'sod', 'pot']

    # # Check if any required key has a None value
    # for key in required_keys:
    #     if data.get(key) is None:
    #         st.write("Please enter all the required values to proceed with the simulation.")
    #         raise ValueError(f"The value for {key} cannot be None.")

    
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

    # gfr
    st.session_state['gfr_values'] = {
            "stage 1": [90.0, 120.0, 200.0],
            "stage 2": [40.0, 75.0, 110.0],
            "stage 3": [10.0, 45.0, 80.0],
            "stage 4": [0.0, 22.0, 40.0],
            "stage 5": [0.0, 0.0, 20.0]
    }

    # bun
    st.session_state['bun_values'] = {
            "very low": [0.0, 0.0, 4.0, 10.0],
            "low": [5.0, 15.0, 25.0],
            "medium": [20.0, 30.0, 40.0],
            "high": [35.0, 50.0, 65.0],
            "very high": [60.0, 80.0, 100.0, 100.0]
    }

    # creatinine
    st.session_state['creatinine_values'] = {
            "low": [0.0, 0.0, 0.7],
            "normal": [0.6 , 0.9, 1.5],
            "medium": [1.3, 1.7, 2.3],
            "high": [2.0, 3.0, 5.0, 5.0]
    }
    # hemoglobin
    st.session_state['hemoglobin_values'] = {
            "low": [5.0, 5.0, 10.0, 13.0],
            "normal": [11.0, 13.75, 16.5],
            "high": [14.5, 17.0, 20.0, 20.0]
    }

    st.session_state['bp_values'] = {
            "normal": [50.0, 50.0, 90.0, 135.0],
            "high": [105.0, 130.0, 155.0],
            "very high": [125.0, 160.0, 200.0, 200.0]
    }

    st.session_state['albuminuria_values'] = {
            "normal": [0.0, 0.0, 1.0],
            "trace": [0.0, 1.0, 2.0],
            "low": [1.0, 2.0, 3.0],
            "medium": [2.0, 3.0, 4.0],
            "high": [3.0, 4.0, 5.0],
            "very high": [4.0, 5.0, 5.0]
    }


    st.session_state['sodium_values'] = {
        "low": [120.0, 120.0, 132.5, 137.0],
        "normal": [133.0, 140.0, 147.0],
        "high": [145.0, 145.5, 150.0, 150.0]
    }

    st.session_state['potassium_values'] = {
        "low": [2.0, 2.0, 3.25, 5.0],
        "normal": [3.0, 4.25, 5.5],
        "high": [4.5, 5.25, 7.0, 7.0]
    }

    st.session_state['severity_values'] = {
        "low": [0.0, 0.0, 0.5],
        "medium": [0.0, 0.5, 1.0],
        "high": [0.5, 1.0, 1.0]
    }

    st.session_state

    for key in frame:
        if key != 'al' and key != 'su':
            if frame[key][0] == 0:
                frame[key] = [np.nan]
    data = pd.DataFrame(frame)
    
    # Convert non-numeric placeholders to NaN
    data.replace({'\t?': np.nan, '?': np.nan, '-': np.nan}, inplace=True)


    
    try:
        history = pd.read_csv('history.csv')
        history = pd.concat([history, data], ignore_index=True)
    except FileNotFoundError:
        history = data

    
    history.to_csv('history.csv', index=False)

    
    st.session_state["new_data"] = data

   
    st.switch_page("pages/3_Check Result.py")


