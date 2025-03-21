from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from model_cnn import create_cnn
from model_mtnn import create_mtnn
from data_preprocessing import preprocess_data
from load_data import load_multiomics_data

# Load dataset
df = load_multiomics_data()

# Extract features and labels
X = df[['Genomic_Feature_1', 'Genomic_Feature_2', 'Genomic_Feature_3', 'Genomic_Feature_4', 'Genomic_Feature_5']].values
y_resistance = df['Resistance_Label'].values
y_response = df['Drug_Response_Probability'].values

# Train models
def train_models(X, y_resistance, y_response):
    X_train, X_test, y_train_res, y_test_res = train_test_split(X, y_resistance, test_size=0.2, random_state=42)
    X_train, X_test, y_train_resp, y_test_resp = train_test_split(X, y_response, test_size=0.2, random_state=42)
    
    cnn_model = create_cnn((X.shape[1], 1))
    cnn_model.fit(X_train, y_train_res, epochs=20, batch_size=32, validation_data=(X_test, y_test_res), callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
    
    mtnn_model = create_mtnn((X.shape[1],))
    mtnn_model.fit(X_train, {'resistance': y_train_res, 'drug_response': y_train_resp}, epochs=20, batch_size=32, validation_data=(X_test, {'resistance': y_test_res, 'drug_response': y_test_resp}), callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
    
    return cnn_model, mtnn_model

# Train models with loaded dataset
cnn_model, mtnn_model = train_models(X, y_resistance, y_response)
