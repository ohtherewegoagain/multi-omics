from flask import Flask, request, jsonify
import numpy as np
from model_cnn import create_cnn
from predict_patient_resistance import predict_resistance

app = Flask(__name__)
model = create_cnn((100, 1))  # Example shape

@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['features']).reshape(1, -1)
    result = predict_resistance(model, data)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
