
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
    new_data['gfr'] = new_data.apply(calculate_gfr, axis=1)
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
new_data = pd.DataFrame({
    'age': [48],
    'bp': [80],
    'sg': [1.02],
    'al': [0],
    'su': [0],
    'bgr': [121],
    'bu': [36],
    'sc': [1],
    'sod': [137],
    'pot': [4.4],
    'hemo': [15.4],
    'pcv': [44],
    'wc': [7800],
    'rc': [5.2],
    'rbc': ['normal'],
    'pc': ['normal'],
    'pcc': ['notpresent'],
    'ba': ['notpresent'],
    'htn': ['yes'],
    'dm': ['no'],
    'cad': ['no'],
    'appet': ['good'],
    'pe': ['no'],
    'ane': ['no'], 
})

prediction, probability = predict_with_knn(knn_best, new_data)
print(f'Prediction: {prediction}')
print(f'Probability: {probability}')

# Define fuzzy variables
prob_ckd = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_ckd')
decision = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'decision')

# Define fuzzy membership functions
prob_ckd['low'] = fuzz.trimf(prob_ckd.universe, [0, 0, 0.5])
prob_ckd['medium'] = fuzz.trimf(prob_ckd.universe, [0, 0.5, 1])
prob_ckd['high'] = fuzz.trimf(prob_ckd.universe, [0.5, 1, 1])

decision['no_ckd'] = fuzz.trimf(decision.universe, [0, 0, 0.5])
decision['uncertain'] = fuzz.trimf(decision.universe, [0, 0.5, 1])
decision['ckd'] = fuzz.trimf(decision.universe, [0.5, 1, 1])

# Define fuzzy rules
rule1 = ctrl.Rule(prob_ckd['low'], decision['no_ckd'])
rule2 = ctrl.Rule(prob_ckd['medium'], decision['uncertain'])
rule3 = ctrl.Rule(prob_ckd['high'], decision['ckd'])

# Create control system and simulation
decision_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
decision_sim = ctrl.ControlSystemSimulation(decision_ctrl)