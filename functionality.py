
import numpy as np
import skfuzzy as fuzz
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from skfuzzy import control as ctrl

# Calculate GFR using MDRD formula
def calculate_gfr(row):
    scr = row['sc']
    age = row['age']
    gfr = 175 * (scr ** -1.154) * (age ** -0.203)
    return gfr

#calculate bun from bu
def calculate_bun(row): 
    bu = row['bu']
    return bu/2.14

# https://athenslab.gr/en/diagnostikes-exetaseis/blood-urea-nitrogen-13


def load_and_preprocess_data (filepath):

        # Load dataset
    data = pd.read_csv(filepath)
    # Encode target labels

    # Identify categorical and numerical columns
    categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    numerical_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]

    # check for tabs inconsistency
    label_encoder = LabelEncoder()
    data['classification'] = data['classification'].str.strip()
    data['classification'] = label_encoder.fit_transform(data['classification'])

    for col in categorical_cols:
        data[col] = data[col].str.strip()

    data['gfr'] = data.apply(calculate_gfr, axis=1)
    numerical_cols.append('gfr')

    data['bun'] = data.apply(calculate_gfr, axis=1)
    numerical_cols.append('bun')
    
    # Select features and target
    features = data.drop(columns=['id', 'classification'])  # X
    target = data['classification']  # y

    # Convert non-numeric placeholders to NaN
    features.replace({'\t?': np.nan, '?': np.nan}, inplace=True)

    # Define the preprocessing for numerical and categorical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return features, target, preprocessor

def train_knn_model(x_train, y_train, preprocessor):
    # Define the model
    knn = KNeighborsClassifier()

    # Create and evaluate the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', knn)])

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'classifier__n_neighbors': range(1, 10),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    knn_best = grid_search.best_estimator_

    return knn_best, best_params


def predict_with_knn(model, new_data):
    #use model for make prediction
    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)
    return prediction, probability



# Load and preprocess data
file_path = 'dataset/kidney_disease.csv'
features, target, preprocessor = load_and_preprocess_data(file_path)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target)

# Train the KNN model
knn_best, best_params = train_knn_model(x_train, y_train, preprocessor)

# Evaluate the model
y_pred = knn_best.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Predict probabilities for analysis
y_pred_probabilities = knn_best.predict_proba(x_test)

print(f'Best parameters: {best_params}')
print(f'Accuracy: {accuracy}')
print(f'Predicted Labels: {y_pred}')
# print(f'Predicted Probabilities: {y_pred_probabilities}')
# Predict with new data


# Define the universe of discourse for each clinical parameter
gfr = ctrl.Antecedent(np.arange(0, 150, 1), 'gfr')
creatinine = ctrl.Antecedent(np.arange(0, 4, 0.1), 'creatinine')
bun = ctrl.Antecedent(np.arange(0, 200, 1), 'bun')
albuminuria = ctrl.Antecedent(np.arange(0, 10, 0.1), 'albuminuria')
bp = ctrl.Antecedent(np.arange(50, 200, 1), 'bp')
hemoglobin = ctrl.Antecedent(np.arange(5, 20, 0.1), 'hemoglobin')
sodium = ctrl.Antecedent(np.arange(120, 150, 0.1), 'sodium')
potassium = ctrl.Antecedent(np.arange(2, 7, 0.1), 'potassium')

# Output severity
severity = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'severity')

# Membership functions for GFR
# Stage 1: eGFR > 90 mL/min/1.73 m²
gfr['stage 1'] = fuzz.trimf(gfr.universe, [90, 120, 200])

# Stage 2: 60 < eGFR < 89 mL/min/1.73 m²
gfr['stage 2'] = fuzz.trimf(gfr.universe, [40, 75, 110])

# Stage 3: 30 < eGFR < 59 mL/min/1.73 m²
gfr['stage 3'] = fuzz.trimf(gfr.universe, [10, 45, 80])

# Stage 4: 15 < eGFR < 29 mL/min/1.73 m²
gfr['stage 4'] = fuzz.trimf(gfr.universe, [0, 22, 40])

# Stage 5: eGFR < 15 mL/min/1.73 m²
gfr['stage 5'] = fuzz.trimf(gfr.universe, [0, 0, 20])


# # Membership functions for Serum Creatinine

# Very Low: < 0.6 mg/dL
creatinine['low'] = fuzz.trimf(creatinine.universe, [0, 0, 0.7])

# Normal: 0.6 - 1.2 mg/dL
creatinine['normal'] = fuzz.trimf(creatinine.universe, [0.6 , 0.9, 1.5])

# Moderate Increase: 1.3 - 1.9 mg/dL
creatinine['medium'] = fuzz.trimf(creatinine.universe, [1.3, 1.7, 2.3])

# High: > 2.0 mg/dL
creatinine['high'] = fuzz.trapmf(creatinine.universe, [2.0, 3.0, 5.0, 5.0])


# Membership functions for BUN
bun["very low"] = fuzz.trapmf(bun.universe, [0, 0, 4, 10])

# Normal BUN levels (approximate range)
bun['low'] = fuzz.trimf(bun.universe, [5, 15, 25])

# Mildly elevated BUN levels
bun['medium'] = fuzz.trimf(bun.universe, [20, 30, 40])

# Moderately elevated BUN levels
bun['high'] = fuzz.trimf(bun.universe, [35, 50, 65])

# Severely elevated BUN levels
bun['very high'] = fuzz.trapmf(bun.universe, [60, 80, 100, 100])

# # https://athenslab.gr/en/diagnostikes-exetaseis/blood-urea-nitrogen-13


# # Membership functions for Albuminuria
# albuminuria['low'] = fuzz.trapmf(albuminuria.universe, [0, 0, 0.3, 1])
# albuminuria['medium'] = fuzz.trimf(albuminuria.universe, [0.3, 1, 3])
# albuminuria['high'] = fuzz.trapmf(albuminuria.universe, [1, 3, 10, 10])

# # Membership functions for Blood Pressure
# bp['normal'] = fuzz.trapmf(bp.universe, [50, 50, 90, 120])
# bp['high'] = fuzz.trimf(bp.universe, [90, 120, 180])
# bp['very_high'] = fuzz.trapmf(bp.universe, [120, 180, 200, 200])

# # Membership functions for Hemoglobin
# hemoglobin['low'] = fuzz.trapmf(hemoglobin.universe, [5, 5, 10, 12])
# hemoglobin['normal'] = fuzz.trimf(hemoglobin.universe, [10, 12, 16])
# hemoglobin['high'] = fuzz.trapmf(hemoglobin.universe, [12, 16, 20, 20])

# # Membership functions for Sodium
# sodium['low'] = fuzz.trapmf(sodium.universe, [120, 120, 135, 138])
# sodium['normal'] = fuzz.trimf(sodium.universe, [135, 138, 145])
# sodium['high'] = fuzz.trapmf(sodium.universe, [138, 145, 150, 150])

# # Membership functions for Potassium
# potassium['low'] = fuzz.trapmf(potassium.universe, [2, 2, 3.5, 4])
# potassium['normal'] = fuzz.trimf(potassium.universe, [3.5, 4, 5.5])
# potassium['high'] = fuzz.trapmf(potassium.universe, [4, 5.5, 7, 7])

# # Membership functions for Severity
# severity['low'] = fuzz.trimf(severity.universe, [0, 0, 0.5])
# severity['medium'] = fuzz.trimf(severity.universe, [0, 0.5, 1])
# severity['high'] = fuzz.trimf(severity.universe, [0.5, 1, 1])

# # Define fuzzy rules for CKD severity
# rule1 = ctrl.Rule(creatinine['high'] & bun['high'] & albuminuria['high'], severity['high'])
# rule2 = ctrl.Rule(creatinine['medium'] & bun['medium'] & albuminuria['medium'], severity['medium'])
# rule3 = ctrl.Rule(creatinine['low'] & bun['low'] & albuminuria['low'], severity['low'])
# # Add more rules as necessary


# # Create the control system and simulation
# severity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
# severity_simulation = ctrl.ControlSystemSimulation(severity_ctrl)


# def analyze_severity_with_fuzzy(new_data, knn_model):

#     #add new gfr, bun column for analysis
#     new_data['gfr'] = new_data.apply(calculate_gfr, axis=1)
#     new_data['bun'] = new_data.apply(calculate_bun, axis=1)
    
#     # Predict with KNN model
#     prediction, probability = predict_with_knn(knn_model, new_data)

#     # Fuzzy analysis for severity
#     # severity_simulation.input['gfr'] = new_data['gfr'].values[0]
#     severity_simulation.input['creatinine'] = new_data['sc'].values[0]
#     severity_simulation.input['bun'] = new_data['bu'].values[0]
#     severity_simulation.input['albuminuria'] = new_data['al'].values[0]
#     # severity_simulation.input['bp'] = new_data['bp'].values[0]
#     # severity_simulation.input['hemoglobin'] = new_data['hemo'].values[0]
#     # severity_simulation.input['sodium'] = new_data['sod'].values[0]
#     # severity_simulation.input['potassium'] = new_data['pot'].values[0]

#     # Compute the fuzzy output
#     severity_simulation.compute()
#     severity_result = severity_simulation.output['severity']

#     return prediction, probability, severity_result



# new_data = pd.DataFrame({
#     'age': [48],
#     'bp': [80],
#     'sg': [1.02],
#     'al': [0],
#     'su': [0],
#     'bgr': [121],
#     'bu': [36],
#     'sc': [1],
#     'sod': [137],
#     'pot': [4.4],
#     'hemo': [15.4],
#     'pcv': [44],
#     'wc': [7800],
#     'rc': [5.2],
#     'rbc': ['normal'],
#     'pc': ['normal'],
#     'pcc': ['notpresent'],
#     'ba': ['notpresent'],
#     'htn': ['yes'],
#     'dm': ['no'],
#     'cad': ['no'],
#     'appet': ['good'],
#     'pe': ['no'],
#     'ane': ['no'], 
# })

# # prediction, probability = predict_with_knn(knn_best, new_data)
# # print(f'Prediction: {prediction}')
# # print(f'Probability: {probability}')


# prediction, probability, severity = analyze_severity_with_fuzzy(new_data, knn_best)
# print(f'Prediction: {prediction}')
# print(f'Probability: {probability}')
# print(f'Severity: {severity}')
