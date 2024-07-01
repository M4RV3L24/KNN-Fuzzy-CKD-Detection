import streamlit as st
import pandas as pd
import os


st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    """, unsafe_allow_html=True)


st.markdown("""
    <h1 style='text-align:  center; color: black;'>📝 Patients's Data History </h1>
    """, unsafe_allow_html=True)

# Function to delete a specific row in the history file
def delete_row(index):
    try:
        history = pd.read_csv('history.csv')
        history.drop(index, inplace=True)
        history.to_csv('history.csv', index=False)
        st.success(f"Patient {index+1}'s data has been deleted.")
        st.experimental_rerun()  # Rerun the app to reflect changes
    except FileNotFoundError:
        st.warning("No history file found to delete data from.")

# Attempt to read the history CSV file
try:
    history = pd.read_csv('history.csv')

    # Iterate through each row in the dataframe
    for i, row in history.iterrows():
        st.write(f"Patient {i+1}")  
        st.write(row.to_frame().T)
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Select Patient {i+1} for Diagnosis", key=f"select_button_{i}"):
                frame = row.to_frame().T  # Convert the row to a DataFrame
                st.session_state["new_data"] = frame  # Store the DataFrame in session state
                st.session_state["selected_data"] = frame  # Store the DataFrame in session state for selected data
                st.switch_page("pages/3_Check Result.py")
        with col2:
            if st.button(f"Delete Patient {i+1} Data", key=f"delete_button_{i}"):
                delete_row(i)

except FileNotFoundError:
    # Handle the case where the history file is not found
    st.write("No history available. Please add some patient data first.")
