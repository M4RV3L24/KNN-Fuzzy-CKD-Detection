import numpy as np
import skfuzzy as fuzz
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# id: Identifier - A unique number assigned to each patient.
# age: Age - The age of the patient in years.
# bp: Blood Pressure - The patient's blood pressure, typically measured in mmHg.
# sg: Specific Gravity - A measure of the concentration of solutes in the urine.
# al: Albumin - The presence of albumin in the urine, an indicator of kidney function.
# su: Sugar - The presence of sugar in the urine, an indicator of diabetes.
# rbc: Red Blood Cells - The count of red blood cells in the urine.
# pc: Pus Cell - The count of pus cells in the urine, indicating infection.
# pcc: Pus Cell Clumps - The presence of clumps of pus cells in the urine.
# ba: Bacteria - The presence of bacteria in the urine.
# bgr: Blood Glucose Random - Random blood glucose level.
# bu: Blood Urea - The level of urea in the blood, an indicator of kidney function.
# sc: Serum Creatinine - The level of creatinine in the blood, an indicator of kidney function.
# sod: Sodium - The level of sodium in the blood.
# pot: Potassium - The level of potassium in the blood.
# hemo: Hemoglobin - The level of hemoglobin in the blood.
# pcv: Packed Cell Volume - The volume percentage of red blood cells in blood.
# wc: White Blood Cell Count - The count of white blood cells in the blood.
# rc: Red Blood Cell Count - The count of red blood cells in the blood.
# htn: Hypertension - Indicates if the patient has hypertension (1 if yes, 0 if no).
# dm: Diabetes Mellitus - Indicates if the patient has diabetes (1 if yes, 0 if no).
# cad: Coronary Artery Disease - Indicates if the patient has coronary artery disease (1 if yes, 0 if no).
# appet: Appetite - The patient's appetite (1 if good, 0 if poor).
# pe: Pedal Edema - Indicates if the patient has pedal edema (1 if yes, 0 if no).
# ane: Anemia - Indicates if the patient has anemia (1 if yes, 0 if no).
# classification: Classification - The classification of the patient's condition (0 if not chronic kidney disease, 1 if chronic kidney disease).

# Load dataset
data = pd.read_csv('dataset/kidney_disease.csv')

