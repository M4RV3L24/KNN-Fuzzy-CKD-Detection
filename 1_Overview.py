import streamlit as st
import functionality as f
import pandas as pd
import numpy as np


st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    """, unsafe_allow_html=True)

def unauthenticated_menu():
    # Show a navigation menu for unauthenticated users
    st.sidebar.page_link("app.py", label="Overview")
    st.sidebar.page_link("pages/result.py", label="Check Result")
    st.sidebar.page_link("pages/history.py", label="Input History")


# @st.cache_data
def load_data():
    data = pd.read_csv('dataset/kidney_disease.csv')
    return data

def display_table(dataframe, page, items_per_page):
    start_idx = page*items_per_page
    end_idx = start_idx + items_per_page
    return dataframe[start_idx:end_idx]
    

st.markdown("""
    <h1 style='text-align:  center; '>Chronic Kidney Disease Analyzer </h1>
    """, unsafe_allow_html=True)


st.markdown('<i class="fa fa-info-circle"></i> Informasi penting tentang CKD', unsafe_allow_html=True)

section1, section2 = st.columns(2)
section1_exp = section1.expander("Expand")

section1_exp.write("""
Chronic Kidney Disease (CKD) is a significant health issue that affects millions of people worldwide. Early detection and diagnosis are crucial for effective management and treatment of the disease. Traditional methods for diagnosing CKD can be time-consuming and require extensive medical expertise. With advancements in machine learning and artificial intelligence, it's now possible to leverage these technologies to assist in the early detection of CKD based on lab results from blood and urine tests.
""")

section2_exp = section2.expander("Expand") 
section2_exp.write("""
This project aims to develop a medical application that can pre-diagnose CKD using a combination of K-Nearest Neighbors (KNN) algorithm and fuzzy logic. The application will analyze lab results, specifically blood and urine test parameters, to identify potential CKD cases. This tool can serve as an aid for medical professionals, providing them with a quick and reliable preliminary diagnosis.
""")

data_load_state = st.text('Loading dataset...')
data = load_data()
data_load_state.text('Loading data...done!')


# Initialize session state
if 'page_number' not in st.session_state:
    st.session_state["page_number"] = 0

  

st.sidebar.html("<b>Table Setting</b>")


# Number of items per page
items_per_page = st.sidebar.number_input("Items per page", min_value=1, max_value=20, value=10, step=1)



# Current page number
total_pages = len(data) // items_per_page
if len(data) % items_per_page != 0:
    total_pages += 1


# Display the table
st.write(f"Displaying page {st.session_state['page_number'] + 1} of {total_pages}")
st.table(display_table(data, st.session_state['page_number'], items_per_page))
# Navigation buttons
col1, col2, col3= st.columns([1, 2, 1], gap="small")

prev, next = col3.columns(2)
with prev:
    if st.button("Prev"):
        if st.session_state["page_number"] > 0:
            st.session_state["page_number"] -= 1
with next:
    if st.button("Next"):
        if st.session_state["page_number"] < total_pages - 1:
            st.session_state["page_number"] += 1

with col2:
    st.write("")
    
with col1:
    st.download_button("Download CSV", data.to_csv().encode("utf-8"), "dataset.csv", "text/csv")


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

