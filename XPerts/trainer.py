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
        self.X = X
        self.y = y
        self.model = None


    def initialize_model(self):
        # model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
        # model.trainable = False
        # flatten_layer = layers.Flatten()
        # dense_layer_1= layers.Dense(200,activation='relu')
        # drop = layers.Dropout(0.3)
        # dense_layer_2= layers.Dense(100,activation='relu')
        # dense_layer_3= layers.Dense(50,activation='relu')
        # dense_layer_4= layers.Dense(25,activation='relu')
        # prediction_layer = layers.Dense(15, activation='linear')
        # self.model = Sequential([model,flatten_layer,
        #                          drop,
        #                          dense_layer_1,
        #                          drop,dense_layer_2,
        #                          dense_layer_3,
        #                          dense_layer_4,
        #                          prediction_layer])
        # nadam_opt = tf.keras.optimizers.Nadam(learning_rate=0.001)
        # self.model.compile(loss='mse',
        #                    optimizer=nadam_opt)
        self.model = Sequential()
        self.model.add(tf.keras.layers.Conv2D(16, (4,4), activation='relu',input_shape=(512,512,1)))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(5,5)))
        self.model.add(tf.keras.layers.Conv2D(8, (3,3),activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(4,4)))
        self.model.add(tf.keras.layers.Conv2D(4, (3,3),activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))
        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                    metrics=['accuracy'],
                    optimizer='adam'
                    )
        return self

    def fit_model(self):
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(self.X,self.y, epochs=100, batch_size=8, validation_split=0.2 ,callbacks=[es])
        return self


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
    # trainer.save_model_locally()
    trainer.save_model_to_gcp()
