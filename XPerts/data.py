import pandas as pd
import numpy as np
from google.cloud import storage
import tensorflow as tf
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH
from os.path import dirname, abspath, join
from dotenv import load_dotenv
import os



env_path = join(dirname(abspath(__file__)),'.env') # ../.env
load_dotenv(env_path)
csv_path = './XPerts/data/data.csv'
data_path = './XPerts/data'


def get_X_from_gcp():
    storage_client = storage.Client.from_service_account_json(os.getenv("gcp_json_path"))
    bucket = storage_client.bucket(BUCKET_NAME)

    num = [*range(1,590)]

    for i in num:
        blob = bucket.blob(f'{BUCKET_TRAIN_X_PATH}/{i}.jpg')
        img = blob.download_to_filename(f'{data_path}/{i}.jpg')


def X_to_tensor():
    num = [*range(1,2)]
    image = []

    for i in num:
        img = tf.keras.utils.load_img(f"{data_path}/{i}.jpg")
        image.append(tf.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0))
    X = tf.concat(image, 0)
    return X


def get_y_from_gcp():
    storage_client = storage.Client.from_service_account_json(os.getenv("gcp_json_path"))
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(BUCKET_TRAIN_y_PATH)
    data = blob.download_to_filename(csv_path)

def y_to_tensor():
    data = pd.read_csv(csv_path)
    data = data.drop(columns='Unnamed: 2')
    data['Nu_Teeth'] = data['Nu_Teeth'].replace(np.nan,'-1')
    data['Nu_Teeth'] = data['Nu_Teeth'].replace('B','-1')
    y = data['Nu_Teeth']
    y_cat = tf.keras.utils.to_categorical(y, num_classes=44)
    y_cat = tf.convert_to_tensor(y_cat)
    return y_cat

if __name__ == '__main__':
    X = X_to_tensor()
    y_cat = y_to_tensor()
    print(y_cat)
