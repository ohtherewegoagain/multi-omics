def predict_resistance(model, patient_data):
    prediction = model.predict(patient_data)
    return "Resistant" if prediction > 0.5 else "Susceptible"
