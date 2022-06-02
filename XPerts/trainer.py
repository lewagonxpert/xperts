import joblib
from google.cloud import storage
from XPerts.data import get_X_from_gcp, get_y_from_gcp,X_to_tensor,y_to_tensor
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np



class Trainer(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None

    def initialize_model(self):
        self.model = Sequential()
        self.model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu',input_shape=(940,2041,3)))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(4,4)))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(4,4)))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(4,4)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(Dense(44, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        return self

    def fit_model(self):
        self.model.fit(self.X,self.y,epochs=5,batch_size=32)
        return self

    def evaluate(self, X_test):
        self.model.predict(X_test)
        return self

    def save_model_locally(self):
        joblib.dump(self.model, 'model.joblib')


if __name__ == "__main__":
    get_X_from_gcp()
    X = X_to_tensor()
    get_y_from_gcp()
    y_cat = y_to_tensor()
    X_train =X[:479]
    X_test = X[479:]
    y_train =y_cat[:479]
    y_test = y_cat[479:]
    trainer = Trainer(X_train, y_train)
    trainer = trainer.initialize_model()
    trainer = trainer.fit_model()
    trainer.evaluate(X_test)
    trainer.save_model_locally()
