import joblib
from google.cloud import storage

from XPerts.data import get_X_from_gcp,get_y_from_gcp,X_to_tensor,get_xml
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH

import tensorflow as tf
from tensorflow.keras.applications import resnet50

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


import pandas as pd
import numpy as np



class Trainer(object):
    def __init__(self, X, y):
        self.X = tf.keras.applications.vgg19.VGG19.preprocess_input(X)
        self.y = y
        self.model = None


    def initialize_model(self):
        model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
        model.trainable = False
        flatten_layer = layers.Flatten()
        dense_layer_1= layers.Dense(200,activation='relu')
        drop = layers.Dropout(0.3)
        dense_layer_2= layers.Dense(100,activation='relu')
        prediction_layer = layers.Dense(15, activation='linear')
        self.model = Sequential([model,flatten_layer,dense_layer_1,drop,dense_layer_2,prediction_layer])
        self.model.compile(loss='mse',
                           optimizer='Nadam')
        return self

    def fit_model(self):
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(self.X,self.y, epochs=100, batch_size=16, validation_split=0.2 ,callbacks=[es])
        return self

    # def evaluate(self, X_test, y_test):
    #     self.model.evaluate(X_test, y_test)
    #     return self


    def save_model_locally(self):
        self.model.save('model.h5')

    def save_model_to_gcp(self):
        local_model_name = 'model.h5'
        self.model.save(local_model_name)
        client = storage.Client().bucket(BUCKET_NAME)
        storage_location = f"model/xperts/v1/{local_model_name}"
        blob = client.blob(storage_location)
        blob.upload_from_filename(local_model_name)




if __name__ == "__main__":
    res = get_X_from_gcp()
    X = X_to_tensor(res)
    y = get_xml()

    X_train =X[:1000]
    X_test = X[1000:]
    y_train =y[:1000]
    y_test = y[1000:]
    trainer = Trainer(X_train, y_train)
    trainer = trainer.initialize_model()
    trainer = trainer.fit_model()
    # trainer.evaluate(X_test, y_test)
    # trainer.save_model_locally()
    trainer.save_model_to_gcp()
