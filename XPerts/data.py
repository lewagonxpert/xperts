import pandas as pd
import numpy as np
from google.cloud import storage
import tensorflow as tf
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH, BUCKET_TRAIN_y_PATH

#get data from GCP
def get_X_from_gcp():
    client = storage.Client()
    num = [*range(1,399)]
    image = []
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_X_PATH}"
    for i in num:
        img = tf.keras.utils.load_img(f"path/{i}.jpg")
        image.append(tf.expand_dims(tf.keras.preprocessing.imageimg_to_array(img), axis=0))
    X= tf.concat(image, 0)
    return X

def get_y_from_gcp():
    client = storage.Client()
    data = pd.read_csv('BUCKET_TRAIN_y_PATH')
    data=data.drop(columns='Unnamed: 2')
    data['Nu_Teeth']=data['Nu_Teeth'].replace(np.nan,'-1')
    data['Nu_Teeth']=data['Nu_Teeth'].replace('B','-1')
    y = data['Nu_Teeth']
    y_cat = tf.keras.utilsto_categorical(y, num_classes=44)
    y_cat = tf.convert_to_tensor(y_cat)
    return y_cat
