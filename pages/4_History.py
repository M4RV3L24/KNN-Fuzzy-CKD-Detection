import streamlit as st
import pandas as pd
import os

st.header("Patient's Data History")

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

    # Display the data in a table
    st.write("Here is the patient's data history:")

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
