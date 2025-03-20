from flask import Flask, request, jsonify
import numpy as np
from predict_patient_resistance import predict_resistance

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict antibiotic resistance."""
    data = np.array(request.json['features']).reshape(1, -1)
    result = predict_resistance(data)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
