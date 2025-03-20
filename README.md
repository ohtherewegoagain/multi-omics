# AI-Enhanced Multi-Omics Framework

## Overview
This repository contains an AI-driven multi-omics framework for predicting chemotherapy-induced antibiotic resistance in pediatric cancer patients. The framework integrates convolutional neural networks (CNNs) and multi-task neural networks (MTNNs) to analyze bacterial resistance patterns and patient-specific drug responses.

## Features
- **Deep Learning Models**: CNN for bacterial resistance classification, MTNN for patient-specific drug response prediction.
- **Multi-Omics Data Integration**: Uses genomic, transcriptomic, and proteomic data.
- **Explainable AI**: SHAP analysis for feature importance.
- **Visualization Tools**: Heatmaps, ROC curves, and SHAP plots.
- **Clinical Integration**: Flask API for real-time predictions.

## File Structure
```bash
📂 ai-multiomics-framework
│── model_cnn.py            # CNN model for bacterial resistance prediction
│── model_mtnn.py           # MTNN for resistance & drug response prediction
│── data_preprocessing.py   # Data cleaning, imputation, and normalization
│── load_data.py            # Loads multi-omics dataset
│── train_models.py         # Trains CNN and MTNN models
│── evaluate_models.py      # Evaluates model performance (AUROC, precision, recall)
│── visualizations.py       # Generates heatmaps and ROC curves
│── predict_patient_resistance.py  # Predicts antibiotic resistance for new patient data
│── clinical_dashboard.py   # Flask API for real-time clinical predictions
│── requirements.txt        # List of dependencies
│── README.md               # Documentation
```

## Installation
```bash
git clone https://github.com/yourusername/ai-multiomics-framework.git
cd ai-multiomics-framework
pip install -r requirements.txt
```

## Running the Model
```python
from load_data import load_multiomics_data
from train_models import train_models

# Load dataset
df = load_multiomics_data("your_dataset.csv")

# Train models
cnn_model, mtnn_model = train_models(X, y_resistance, y_response)
```

## Evaluating Models
```python
from evaluate_models import evaluate_model
evaluate_model(cnn_model, X_test, y_test_resistance)
```

## Running the Flask API (Optional)
```bash
python clinical_dashboard.py
```
Make a POST request with patient features:
```json
{
    "features": [0.8, 1.2, 0.4, ...]
}
```

## Future Enhancements
- **Expand dataset**: Include more diverse patient and bacterial strain data.
- **Improve explainability**: Integrate more AI interpretability methods.
- **Optimize clinical integration**: Connect with electronic health records (EHRs).

## Contributor
- **Avani Agarwal**

## License
This project is licensed under the MIT License.
