import streamlit as st
import functionality as f
import pandas as pd
import numpy as np



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
    

st.title("Chronic Kidney Disease Analyzer")

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
