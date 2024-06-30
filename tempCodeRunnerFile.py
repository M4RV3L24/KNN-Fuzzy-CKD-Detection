def analyze_severity_with_fuzzy(new_data, knn_model):

    #add new gfr, bun column for analysis
    new_data['gfr'] = new_data.apply(calculate_gfr, axis=1)
    new_data['bun'] = new_data.apply(calculate_bun, axis=1)
    
    # Predict with KNN model
    prediction, probability = predict_with_knn(knn_model, new_data)

    # Fuzzy analysis for severity
    # severity_simulation.input['gfr'] = new_data['gfr'].values[0]
    severity_simulation.input['creatinine'] = new_data['sc'].values[0]
    severity_simulation.input['bun'] = new_data['bu'].values[0]
    severity_simulation.input['albuminuria'] = new_data['al'].values[0]
    # severity_simulation.input['bp'] = new_data['bp'].values[0]
    # severity_simulation.input['hemoglobin'] = new_data['hemo'].values[0]
    # severity_simulation.input['sodium'] = new_data['sod'].values[0]
    # severity_simulation.input['potassium'] = new_data['pot'].values[0]

    # Compute the fuzzy output
    severity_simulation.compute()
    severity_result = severity_simulation.output['severity']

    return prediction, probability, severity_result



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

# prediction, probability = predict_with_knn(knn_best, new_data)
# print(f'Prediction: {prediction}')
# print(f'Probability: {probability}')


prediction, probability, severity = analyze_severity_with_fuzzy(new_data, knn_best)
print(f'Prediction: {prediction}')
print(f'Probability: {probability}')
print(f'Severity: {severity}')