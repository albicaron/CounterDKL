import numpy as np
import keras
from keras import layers


class Autoencoder:

    def __init__(self, input_dim=7, hidden_dim_1=10, hidden_dim_2=5, encoding_dim=2):

        input_layer = keras.Input(shape=(input_dim,))

        # Encoder
        encoded = layers.Dense(hidden_dim_1, activation='linear')(input_layer)
        encoded = layers.Dense(hidden_dim_2, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='linear')(encoded)

        # Decoder
        decoded = layers.Dense(hidden_dim_2, activation='relu')(encoded)
        decoded = layers.Dense(hidden_dim_1, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        # Model
        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)

        self.autoencoder.compile(metrics=['accuracy'],
                                 loss='mean_squared_error',
                                 optimizer='adam')

    def fit(self, x_train, x_val):

        self.autoencoder.fit(x_train, x_train,
                             epochs=500,
                             batch_size=500,
                             shuffle=True,
                             validation_data=(x_val, x_val))

    def encod_pred(self, x_pred):

        return np.array(self.encoder.predict(x_pred))
