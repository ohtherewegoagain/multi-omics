import numpy as np
from load_data import load_multiomics_data
from model_cnn import create_cnn

# Load trained model
cnn_model = create_cnn((5, 1))  # Adjust input shape as per dataset

def predict_resistance(patient_features):
    """Predicts bacterial resistance based on patient genomic features."""
    prediction = cnn_model.predict(np.array(patient_features).reshape(1, -1))
    return "Resistant" if prediction > 0.5 else "Susceptible"
