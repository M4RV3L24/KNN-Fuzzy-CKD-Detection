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
import joblib
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
        'classifier__n_neighbors': range(1, 30),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    knn_best = grid_search.best_estimator_

    # Save the best model and parameters
    joblib.dump(knn_best, 'knn_best_model.pkl')  # Save the model
    joblib.dump(best_params, 'knn_best_params.pkl')  # Save the best parameters

    return knn_best, best_params


def predict_with_knn(model, new_data):
    #use model for make prediction
    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)
    return prediction, probability

def fuzzy_init():
    # Define the universe of discourse for each clinical parameter
    gfr = ctrl.Antecedent(np.arange(0, 300, 1), 'gfr')
    creatinine = ctrl.Antecedent(np.arange(0, 4, 0.1), 'creatinine')
    bun = ctrl.Antecedent(np.arange(0, 100, 1), 'bun')
    albuminuria = ctrl.Antecedent(np.arange(0, 6, 1), 'albuminuria')
    bp = ctrl.Antecedent(np.arange(50, 200, 1), 'bp')
    hemoglobin = ctrl.Antecedent(np.arange(5, 20, 0.1), 'hemoglobin')
    sodium = ctrl.Antecedent(np.arange(120, 150, 0.1), 'sodium')
    potassium = ctrl.Antecedent(np.arange(2, 7, 0.1), 'potassium')

    # Output severity
    severity = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'severity')

    return {
        "gfr": gfr, 
        "creatinine": creatinine, 
        "bun": bun, 
        "albuminuria": albuminuria,
        "bp": bp,
        "hemoglobin": hemoglobin,
        "sodium": sodium,
        "potassium": potassium,
        "severity": severity
            }

def gfr_member(gfr, value=None, shape="triangle"):
    # Membership functions for GFR
    # check value is null or not
    if value is None:
        value = {
            "stage 1": [90, 120, 500],
            "stage 2": [40, 75, 110],
            "stage 3": [10, 45, 80],
            "stage 4": [0, 22, 40],
            "stage 5": [0, 0, 20]
        }
    stage1 = value["stage 1"]
    stage2 = value["stage 2"]
    stage3 = value["stage 3"]
    stage4 = value["stage 4"]
    stage5 = value["stage 5"]
    gfr['stage 1'] = fuzz.trimf(gfr.universe, stage1)

    # Stage 2: 60 < eGFR < 89 mL/min/1.73 m²
    gfr['stage 2'] = fuzz.trimf(gfr.universe, stage2)

    # Stage 3: 30 < eGFR < 59 mL/min/1.73 m²
    gfr['stage 3'] = fuzz.trimf(gfr.universe, stage3)

    # Stage 4: 15 < eGFR < 29 mL/min/1.73 m²
    gfr['stage 4'] = fuzz.trimf(gfr.universe, stage4)

    # Stage 5: eGFR < 15 mL/min/1.73 m²
    gfr['stage 5'] = fuzz.trimf(gfr.universe, stage5)

    data = {"eGFR": gfr.universe}
    for key, mf in gfr.terms.items():
        data[key] = mf.mf

    df = pd.DataFrame(data)
    st.line_chart(df, x="eGFR", y=["stage 1", "stage 2", "stage 3", "stage 4", "stage 5"], 
                  x_label= "eGFR (mL/min/1.73 m²)", y_label="Membership Degree")

    return gfr


    # # Membership functions for Serum Creatinine
def creatinine_member(creatinine, value=None, shape="triangle"):
    if value is None:
        value = {
            "low": [0, 0, 0.7],
            "normal": [0.6 , 0.9, 1.5],
            "medium": [1.3, 1.7, 2.3],
            "high": [2.0, 3.0, 5.0, 5.0]
        }
    low = value["low"]
    normal = value["normal"]
    medium = value["medium"]
    high = value["high"]
    # Very Low: < 0.6 mg/dL
    creatinine['low'] = fuzz.trimf(creatinine.universe, low)

    # Normal: 0.6 - 1.2 mg/dL
    creatinine['normal'] = fuzz.trimf(creatinine.universe, normal)

    # Moderate Increase: 1.3 - 1.9 mg/dL
    creatinine['medium'] = fuzz.trimf(creatinine.universe, medium)

    # High: > 2.0 mg/dL
    creatinine['high'] = fuzz.trapmf(creatinine.universe, high)

    data = {"creatinine": creatinine.universe}
    for key, mf in creatinine.terms.items():
        data[key] = mf.mf

    df = pd.DataFrame(data)
    st.line_chart(df, x="creatinine", y=["low", "normal", "medium", "high"], 
                  x_label= "SCr (mg/dl)", y_label="Membership Degree")


    return creatinine

def bun_member(bun, value=None, shape="triangle"):
    if value is None:
        value = {
            "very low": [0, 0, 4, 10],
            "low": [5, 15, 25],
            "medium": [20, 30, 40],
            "high": [35, 50, 65],
            "very high": [60, 80, 100, 100]
        }
    very_low = value["very low"]
    low = value["low"]
    medium = value["medium"]
    high = value["high"]
    very_high = value["very high"]

    # Membership functions for BUN
    bun["very low"] = fuzz.trapmf(bun.universe, very_low)

    # Normal BUN levels (approximate range) 7-20 mg/dl
    bun['low'] = fuzz.trimf(bun.universe, low)

    # Mildly elevated BUN levels
    bun['medium'] = fuzz.trimf(bun.universe, medium)

    # Moderately elevated BUN levels
    bun['high'] = fuzz.trimf(bun.universe, high)

    # Severely elevated BUN levels
    bun['very high'] = fuzz.trapmf(bun.universe, very_high)

    data = {"bun": bun.universe}
    for key, mf in bun.terms.items():
        data[key] = mf.mf

    df = pd.DataFrame(data)
    st.line_chart(df, x="bun", y=["very low","low", "medium", "high", "very high"], 
                  x_label= "bun (mg/dl)", y_label="Membership Degree")

    return bun

    # # https://athenslab.gr/en/diagnostikes-exetaseis/blood-urea-nitrogen-13

def albuminuria_member(albuminuria, value=None, shape="triangle"):
    if value is None:
        value = {
            "normal": [0, 0, 1],
            "trace": [0, 1, 2],
            "low": [1, 2, 3],
            "medium": [2, 3, 4],
            "high": [3, 4, 5],
            "very high": [4, 5, 5]
        }
    normal = value["normal"]
    trace = value["trace"]
    low = value["low"]
    medium = value["medium"]
    high = value["high"]
    very_high = value["very high"]

    # Membership functions for Albuminuria (nominal data)
    albuminuria['normal'] = fuzz.trimf(albuminuria.universe, normal)         # < 30 mg/g
    albuminuria['trace'] = fuzz.trimf(albuminuria.universe, trace)          # 30 - 300 mg/g
    albuminuria['low'] = fuzz.trimf(albuminuria.universe, low)            # 300 - 1000 mg/g
    albuminuria['medium'] = fuzz.trimf(albuminuria.universe, medium)         # 1000 - 3000 mg/g
    albuminuria['high'] = fuzz.trimf(albuminuria.universe, high)           # > 3000 mg/g
    albuminuria['very high'] = fuzz.trimf(albuminuria.universe, very_high)      # > 3000 mg/g


    data = {"albuminuria": albuminuria.universe}
    for key, mf in albuminuria.terms.items():
        data[key] = mf.mf
    
    df = pd.DataFrame(data)
    st.line_chart(df, x="albuminuria", y=["normal", "trace", "low", "medium", "high", "very high"], 
                x_label= "Albuminuria (mg/g)", y_label="Membership Degree")
    return albuminuria

def bp_member(bp, value=None, shape="triangle"):
    if value is None:
        value = {
            "normal": [50, 50, 90, 135],
            "high": [105, 130, 155],
            "very high": [125, 160, 200, 200]
        }
    
    normal = value["normal"]
    high = value["high"]
    very_high = value["very high"]
    # Membership functions for Blood Pressure
    bp['normal'] = fuzz.trapmf(bp.universe, normal)     # < 120/80 mmHg
    bp['high'] = fuzz.trimf(bp.universe, high)          # 120/80 - 139/89 mmHg
    bp['very_high'] = fuzz.trapmf(bp.universe, very_high) # ≥ 140/90 mmHg

    data = {"bp": bp.universe}
    for key, mf in bp.terms.items():
        data[key] = mf.mf
    
    df = pd.DataFrame(data)
    st.line_chart(df, x="bp", y=["normal", "high", "very_high"], 
                x_label= "Blood Pressure (mmHg)", y_label="Membership Degree")

    return bp

def hemoglobin_member(hemoglobin, value=None, shape="triangle"):
    if value is None:
        value = {
            "low": [5, 5, 10, 13],
            "normal": [11, 13.75, 16.5],
            "high": [14.5, 17, 20, 20]
        }
    low = value["low"]
    normal = value["normal"]
    high = value["high"]

    # Membership functions for Hemoglobin
    hemoglobin['low'] = fuzz.trapmf(hemoglobin.universe, low)       # < 12 g/dL for women, < 13 g/dL for men
    hemoglobin['normal'] = fuzz.trimf(hemoglobin.universe, normal)  # 12 - 15.5 g/dL for women, 13 - 17.5 g/dL for men
    hemoglobin['high'] = fuzz.trapmf(hemoglobin.universe, high)  # > 15.5 g/dL for women, > 17.5 g/dL for men

    data = {"hemoglobin": hemoglobin.universe}
    for key, mf in hemoglobin.terms.items():
        data[key] = mf.mf
    
    df = pd.DataFrame(data)
    st.line_chart(df, x="hemoglobin", y=["low", "normal", "high"], 
                x_label= "Hemoglobin (g/dL)", y_label="Membership Degree")

    return hemoglobin


def sodium_member(sodium, value=None, shape="triangle"):
    if value is None:
        value = {
            "low": [120, 120, 132.5, 137],
            "normal": [133, 140, 147],
            "high": [145, 145.5, 150, 150]
        }
    low = value["low"]
    normal = value["normal"]
    high = value["high"]
    # Membership functions for Sodium
    sodium['low'] = fuzz.trapmf(sodium.universe, low)   # < 135 mEq/L
    sodium['normal'] = fuzz.trimf(sodium.universe, normal)        # 135 - 145 mEq/L
    sodium['high'] = fuzz.trapmf(sodium.universe, high)  # > 145 mEq/L

    
    data = {"sodium": sodium.universe}
    for key, mf in sodium.terms.items():
        data[key] = mf.mf
    
    df = pd.DataFrame(data)
    st.line_chart(df, x="sodium", y=["low", "normal", "high"], 
                x_label= "Sodium (mEq/L)", y_label="Membership Degree")
    return sodium

def potassium_member(potassium, value=None, shape="triangle"):
    if value is None:
        value = {
            "low": [2, 2, 3.25, 5],
            "normal": [3, 4.25, 5.5],
            "high": [4.5, 5.25, 7, 7]
        }
    low = value["low"]
    normal = value["normal"]
    high = value["high"]

    # Membership functions for Potassium
    potassium['low'] = fuzz.trapmf(potassium.universe, low)    # < 3.5 mEq/L
    potassium['normal'] = fuzz.trimf(potassium.universe, normal)     # 3.5 - 5.0 mEq/L
    potassium['high'] = fuzz.trapmf(potassium.universe, high)     # > 5.0 mEq/L

    data = {"potassium": potassium.universe}
    for key, mf in potassium.terms.items():
        data[key] = mf.mf
    
    df = pd.DataFrame(data)
    st.line_chart(df, x="potassium", y=["low", "normal", "high"],
                x_label= "Potassium (mEq/L)", y_label="Membership Degree")

    return potassium

def severity_member(severity, value=None, shape="triangle"):
    if value is None:
        value = {
            "low": [0, 0, 0.5],
            "medium": [0, 0.5, 1],
            "high": [0.5, 1, 1]
        }
    low = value["low"]
    medium = value["medium"]
    high = value["high"]

    # # Membership functions for Severity
    severity['low'] = fuzz.trimf(severity.universe, low)
    severity['medium'] = fuzz.trimf(severity.universe, medium)
    severity['high'] = fuzz.trimf(severity.universe, high)

    data = {"severity": severity.universe}
    for key, mf in severity.terms.items():
        data[key] = mf.mf
    
    df = pd.DataFrame(data)
    st.line_chart(df, x="severity", y=["low", "medium", "high"],
                x_label= "Severity", y_label="Membership Degree")
    return severity

def create_fuzzy_rules(gfr, creatinine, bun, albuminuria, bp, hemoglobin, sodium, potassium, severity):

    # Define fuzzy rules for CKD severity
    rule1 = ctrl.Rule(gfr['stage 5'] | creatinine['high'] | bun['very high'] | albuminuria['very high'] | 
                    bp['very_high'] | hemoglobin['low'] | sodium['low'] | potassium['high'], severity['high'])

    rule2 = ctrl.Rule((gfr['stage 3'] | gfr['stage 4']) & (creatinine['medium'] | bun['medium'] | albuminuria['medium'] | 
                    bp['high'] | hemoglobin['normal'] | sodium['normal'] | potassium['normal']), severity['medium'])

    rule3 = ctrl.Rule((gfr['stage 1'] | gfr['stage 2']) & (creatinine['low'] | creatinine['normal']) & 
                    (bun['low'] | albuminuria['low'] | albuminuria['normal']) & bp['normal'] & 
                    (hemoglobin['normal'] | hemoglobin['high']) & sodium['normal'] & 
                    (potassium['low'] | potassium['normal']), severity['low'])
    rule4 = ctrl.Rule(gfr['stage 1'], severity["low"])
    rule5 = ctrl.Rule(gfr['stage 2'] | gfr["stage 3"], severity["medium"])
    rule6 = ctrl.Rule(gfr['stage 4'] | gfr["stage 5"], severity["high"])
    rule7 = ctrl.Rule(albuminuria['normal'] | albuminuria["trace"], severity["low"])
    rule8 = ctrl.Rule(albuminuria['low'] | albuminuria["medium"], severity["medium"])
    rule9 = ctrl.Rule(albuminuria['high'] | albuminuria["very high"], severity["high"])
    rule10 = ctrl.Rule(bp['normal'], severity["low"])
    rule11 = ctrl.Rule(bp['high'], severity["medium"])
    rule12 = ctrl.Rule(bp['very_high'], severity["high"])
    rule13 = ctrl.Rule(hemoglobin['normal'] | hemoglobin['high'], severity["low"])
    rule14 = ctrl.Rule(hemoglobin['low'], severity["high"])
    rule15 = ctrl.Rule(sodium['normal'], severity["low"])
    rule16 = ctrl.Rule(sodium['low'], severity["high"])
    rule17 = ctrl.Rule(potassium['normal'] | potassium['low'], severity["low"])
    rule18 = ctrl.Rule(potassium['high'], severity["high"])

    # Create a control system based on the rules
    ckd_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, 
                                   rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18])
    ckd_simulation = ctrl.ControlSystemSimulation(ckd_ctrl)

    return ckd_simulation


# # Input values for the simulation (corresponding to "low" severity)
# ckd_simulation.input['gfr'] = 65  # Example: Stage 2
# ckd_simulation.input['creatinine'] = 1.0  # Example: Normal
# ckd_simulation.input['bun'] = 15  # Example: Low
# ckd_simulation.input['albuminuria'] = 1  # Example: Normal
# ckd_simulation.input['bp'] = 120  # Example: Normal
# ckd_simulation.input['hemoglobin'] = 14  # Example: Normal
# ckd_simulation.input['sodium'] = 140  # Example: Normal
# ckd_simulation.input['potassium'] = 4  # Example: Normal

# # Compute the result
# ckd_simulation.compute()

# # Print the severity level
# print(f"Severity level: {ckd_simulation.output['severity']:.2f}")

# # Visualize the severity graph
# severity.view(sim=ckd_simulation)

# # Save the severity graph to a file
# plt.savefig('severity_graph.png')
# print("Severity graph saved as 'severity_graph.png'")

# image = Image.open('severity_graph.png')
# image.show()


