import pandas as pd
import numpy as np
from google.cloud import storage
import tensorflow as tf
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH
from dotenv import load_dotenv
from PIL import Image
import io
from io import StringIO

def get_X_from_gcp():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    num = [*range(1,599)]
    res = []
    for i in num:
        try:
            blob = bucket.blob(f'{BUCKET_TRAIN_X_PATH}/{i}.jpg')
            test = blob.download_as_bytes()
            res.append(test)
        except:
            print("Couldnt download image")
    return res


def X_to_tensor(res):
    image = []
    for i in res:
        img = Image.open(io.BytesIO(i))
        image.append(np.expand_dims(np.asarray(img),axis=0))
    X = tf.concat(image, 0)
    return X


def get_y_from_gcp():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(BUCKET_TRAIN_y_PATH)
    y = blob.download_as_bytes()
    return y


def y_to_tensor(y):
    s = str(y,'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    df = df.drop(columns='Unnamed: 2')
    df['Nu_Teeth'] = df['Nu_Teeth'].replace(np.nan,'-1')
    df['Nu_Teeth'] = df['Nu_Teeth'].replace('B','-1')
    y = df['Nu_Teeth']
    y_cat = tf.keras.utils.to_categorical(y, num_classes=44)
    y_cat = tf.convert_to_tensor(y_cat)
    return y_cat

if __name__ == '__main__':
    res = get_X_from_gcp()
    X = X_to_tensor(res)
    y = get_y_from_gcp()
    y_cat = y_to_tensor(y)
