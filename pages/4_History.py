import streamlit as st
import pandas as pd

st.header("Patient's Data History")

try:
    # Attempt to read the history CSV file
    history = pd.read_csv('history.csv')

    # Display the data in a table
    st.write("Here is the patient's data history:")

    # Iterate through each row in the dataframe
    for i, row in history.iterrows():
        st.write(f"Patient {i+1}")
        st.write(row.to_frame().T)
        if st.button(f"Select Patient {i+1} for Diagnosis", key=f"button_{i}"):
            frame = row.to_frame().T  # Convert the row to a DataFrame
            st.session_state["new_data"] = frame  # Store the DataFrame in session state
            st.session_state["selected_data"] = frame  # Store the DataFrame in session state for selected data
            st.switch_page("pages/3_Check Result.py")

except FileNotFoundError:
    # Handle the case where the history file is not found
    st.write("No history available. Please add some patient data first.")


