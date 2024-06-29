
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

import numpy as np
import skfuzzy as fuzz
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Load dataset
data = pd.read_csv('dataset/kidney_disease.csv')


# Identify categorical columns (based on the description and data inspection)
categorical_cols = ["rbc", "pc", "pcc",	"ba", "htn", "dm", "cad", "appet", "pe", "ane"]
numerical_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc","rc"]

#select features
#select all column except classification and id
features = data.drop(columns = ['id', 'classification']) #x
target = data['classification'] #y

# Convert non-numeric placeholders to NaN
features.replace({'\t?': np.nan, '?': np.nan}, inplace=True)

# Handling missing values for numerical columns
# Convert features back to DataFrame after imputation to retain column names
int_imputer = SimpleImputer(strategy='mean')# You can choose 'median', 'most_frequent', etc.
for col in numerical_cols:
    features[[col]] = int_imputer.fit_transform(features[[col]])


# Initialize label encoder
le = LabelEncoder()
str_imputer = SimpleImputer(strategy="most_frequent")

# Apply LabelEncoder to each categorical column
for col in categorical_cols:
    # Replace NaN with a placeholder string, as LabelEncoder does not handle NaN
    features[[col]] = str_imputer.fit_transform(features[[col]])
    features[col] = le.fit_transform(features[col])


# Split the data
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target)


# # create and train the model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')


