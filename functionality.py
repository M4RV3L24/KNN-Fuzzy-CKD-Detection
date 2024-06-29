
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

def applyKNN ():

        # Load dataset
    data = pd.read_csv('dataset/kidney_disease.csv')
    # Encode target labels
    label_encoder = LabelEncoder()
    data['classification'] = label_encoder.fit_transform(data['classification'])


    # Identify categorical and numerical columns
    categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    numerical_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]

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

    # Define the model
    knn = KNeighborsClassifier()

    # Create and evaluate the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', knn)])

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'classifier__n_neighbors': range(1, 31),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    knn_best = grid_search.best_estimator_

    # Train and evaluate the model
    knn_best.fit(x_train, y_train)
    y_pred = knn_best.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Best parameters: {best_params}')
    print(f'Accuracy: {accuracy}')
    print(y_pred)
    return y_pred



applyKNN()