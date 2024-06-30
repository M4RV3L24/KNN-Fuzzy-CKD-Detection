import numpy as np
import skfuzzy as fuzz
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


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
    features.replace({'\t?': np.nan, '?': np.nan, '-': np.nan}, inplace=True)

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
bun = ctrl.Antecedent(np.arange(0, 100, 1), 'bun')
albuminuria = ctrl.Antecedent(np.arange(0, 6, 1), 'albuminuria')
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

# Normal BUN levels (approximate range) 7-20 mg/dl
bun['low'] = fuzz.trimf(bun.universe, [5, 15, 25])

# Mildly elevated BUN levels
bun['medium'] = fuzz.trimf(bun.universe, [20, 30, 40])

# Moderately elevated BUN levels
bun['high'] = fuzz.trimf(bun.universe, [35, 50, 65])

# Severely elevated BUN levels
bun['very high'] = fuzz.trapmf(bun.universe, [60, 80, 100, 100])

# # https://athenslab.gr/en/diagnostikes-exetaseis/blood-urea-nitrogen-13


# Membership functions for Albuminuria (nominal data)
albuminuria['normal'] = fuzz.trimf(albuminuria.universe, [0, 0, 1])         # < 30 mg/g
albuminuria['trace'] = fuzz.trimf(albuminuria.universe, [0, 1, 2])          # 30 - 300 mg/g
albuminuria['low'] = fuzz.trimf(albuminuria.universe, [1, 2, 3])            # 300 - 1000 mg/g
albuminuria['medium'] = fuzz.trimf(albuminuria.universe, [2, 3, 4])         # 1000 - 3000 mg/g
albuminuria['high'] = fuzz.trimf(albuminuria.universe, [3, 4, 5])           # > 3000 mg/g
albuminuria['very high'] = fuzz.trimf(albuminuria.universe, [4, 5, 5])      # > 3000 mg/g


# Membership functions for Blood Pressure
bp['normal'] = fuzz.trapmf(bp.universe, [50, 50, 90, 135])     # < 120/80 mmHg
bp['high'] = fuzz.trimf(bp.universe, [105, 130, 155])          # 120/80 - 139/89 mmHg
bp['very_high'] = fuzz.trapmf(bp.universe, [125, 160, 200, 200]) # ≥ 140/90 mmHg


# Membership functions for Hemoglobin
hemoglobin['low'] = fuzz.trapmf(hemoglobin.universe, [5, 5, 10, 13])       # < 12 g/dL for women, < 13 g/dL for men
hemoglobin['normal'] = fuzz.trimf(hemoglobin.universe, [11, 13.75, 16.5])  # 12 - 15.5 g/dL for women, 13 - 17.5 g/dL for men
hemoglobin['high'] = fuzz.trapmf(hemoglobin.universe, [14.5, 17, 20, 20])  # > 15.5 g/dL for women, > 17.5 g/dL for men


# Membership functions for Sodium
sodium['low'] = fuzz.trapmf(sodium.universe, [120, 120, 132.5, 137])   # < 135 mEq/L
sodium['normal'] = fuzz.trimf(sodium.universe, [133, 140, 147])        # 135 - 145 mEq/L
sodium['high'] = fuzz.trapmf(sodium.universe, [145, 145.5, 150, 150])  # > 145 mEq/L


# Membership functions for Potassium
potassium['low'] = fuzz.trapmf(potassium.universe, [2, 2, 3.25, 5])    # < 3.5 mEq/L
potassium['normal'] = fuzz.trimf(potassium.universe, [3, 4.25, 5.5])     # 3.5 - 5.0 mEq/L
potassium['high'] = fuzz.trapmf(potassium.universe, [4.5, 5.25, 7, 7])     # > 5.0 mEq/L


# # Membership functions for Severity
severity['low'] = fuzz.trimf(severity.universe, [0, 0, 0.5])
severity['medium'] = fuzz.trimf(severity.universe, [0, 0.5, 1])
severity['high'] = fuzz.trimf(severity.universe, [0.5, 1, 1])


# Define fuzzy rules for CKD severity
rule1 = ctrl.Rule(gfr['stage 5'] | creatinine['high'] | bun['very high'] | albuminuria['very high'] | 
                  bp['very_high'] | hemoglobin['low'] | sodium['low'] | potassium['high'], severity['high'])

rule2 = ctrl.Rule((gfr['stage 3'] | gfr['stage 4']) & (creatinine['medium'] | bun['medium'] | albuminuria['medium'] | 
                  bp['high'] | hemoglobin['normal'] | sodium['normal'] | potassium['normal']), severity['medium'])

rule3 = ctrl.Rule((gfr['stage 1'] | gfr['stage 2']) & (creatinine['low'] | creatinine['normal']) & 
                  (bun['low'] | albuminuria['low'] | albuminuria['normal']) & bp['normal'] & 
                  (hemoglobin['normal'] | hemoglobin['high']) & sodium['normal'] & 
                  (potassium['low'] | potassium['normal']), severity['low'])

# Create a control system based on the rules
ckd_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
ckd_simulation = ctrl.ControlSystemSimulation(ckd_ctrl)


# Input values for the simulation (corresponding to "low" severity)
ckd_simulation.input['gfr'] = 65  # Example: Stage 2
ckd_simulation.input['creatinine'] = 1.0  # Example: Normal
ckd_simulation.input['bun'] = 15  # Example: Low
ckd_simulation.input['albuminuria'] = 1  # Example: Normal
ckd_simulation.input['bp'] = 120  # Example: Normal
ckd_simulation.input['hemoglobin'] = 14  # Example: Normal
ckd_simulation.input['sodium'] = 140  # Example: Normal
ckd_simulation.input['potassium'] = 4  # Example: Normal

# Compute the result
ckd_simulation.compute()

# Print the severity level
print(f"Severity level: {ckd_simulation.output['severity']:.2f}")

# Visualize the severity graph
severity.view(sim=ckd_simulation)

# Save the severity graph to a file
plt.savefig('severity_graph.png')
print("Severity graph saved as 'severity_graph.png'")

image = Image.open('severity_graph.png')
image.show()