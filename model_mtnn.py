from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model

def create_mtnn(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    output_resistance = Dense(1, activation='sigmoid', name='resistance')(x)
    output_response = Dense(1, activation='linear', name='drug_response')(x)
    
    model = Model(inputs=inputs, outputs=[output_resistance, output_response])
    model.compile(optimizer='adam', loss={'resistance': 'binary_crossentropy', 'drug_response': 'mse'}, metrics=['accuracy'])
    return model
