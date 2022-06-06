import joblib
from google.cloud import storage

from XPerts.data import get_X_from_gcp, get_y_from_gcp,X_to_tensor,y_to_tensor
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, models

import pandas as pd
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



class Trainer(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None


    def initialize_model(self):
        model = tf.keras.applications.vgg16.VGG16(
                                            include_top=False,
                                            weights='imagenet',
                                            input_shape=(940,2041,3),
                                            classes=44,
                                            classifier_activation='softmax'
                                        )
        model.trainable = False

        base_model = model
        flatten_layer = layers.Flatten()
        dense_layer = layers.Dense(500, activation='relu')
        prediction_layer = layers.Dense(44, activation='softmax')


        self.model = models.Sequential([
            base_model,
            flatten_layer,
            dense_layer,
            prediction_layer
        ])


        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        return self

    def fit_model(self):
        self.model.fit(self.X,self.y,epochs=30,batch_size=32)
        return self

    def evaluate(self, X_test, y_test):
        self.model.evaluate(X_test, y_test)
        return self


    def save_model_locally(self):
        joblib.dump(self.model, 'model.joblib')


    def save_model_to_gcp(self):
        local_model_name = 'model.joblib'
        joblib.dump(self.model, local_model_name)
        client = storage.Client().bucket(BUCKET_NAME)
        storage_location = f"model/xperts/v1/{local_model_name}"
        blob = client.blob(storage_location)
        blob.upload_from_filename(local_model_name)



if __name__ == "__main__":
    res = get_X_from_gcp()
    X = X_to_tensor(res)
    y = get_y_from_gcp()
    y_cat = y_to_tensor(y)
    X_train =X[:479]
    X_test = X[479:]
    y_train =y_cat[:479]
    y_test = y_cat[479:]
    trainer = Trainer(X_train, y_train)
    trainer = trainer.initialize_model()
    trainer = trainer.fit_model()
    trainer.evaluate(X_test, y_test)
    trainer.save_model_to_gcp()
