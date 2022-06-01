import joblib
from XPerts.data import X, get_X_from_gcp, get_y_from_gcp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape



class Trainer(object):
    def __init__(self, X, y):
        """
        X: Tensor
        y: Tensor
        """
        self.X = X
        self.y = y
        self.model = None

    def initialize_model(self):
        self.model = Sequential()
        self.model.add(Reshape((940,2041,3), input_shape=(940,2041,3)))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def evaluate(self, X_test):
        self.model.predict(X_test)

    def save_model_locally(self):
        joblib.dump(self.model, 'model.joblib')
