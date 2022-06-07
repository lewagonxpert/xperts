import pandas as pd
import numpy as np
from google.cloud import storage
import tensorflow as tf
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH
<<<<<<< HEAD
from PIL import Image
import io
import xml.etree.ElementTree as ET


def get_X_from_gcp():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)

    image_id =[*range(1,1197)]

    res = []

    for i in image_id:
        try:
            blob = bucket.get_blob(f'{BUCKET_TRAIN_X_PATH}/{i}.jpg')
            test = blob.download_as_bytes()
            res.append(test)
        except Exception as e: print(e)
    return res


def X_to_tensor(res):
    image = []
    for i in res:
        img = Image.open(io.BytesIO(i))
        image.append(np.expand_dims(np.asarray(img),axis=0))
    X = tf.concat(image, 0)
    return X


def convert_box_to_circle(s):
    p1=int((s[0]+s[2])/2)
    p2=int((s[1]+s[3])/2)
    r=(((s[2]-s[0])**2)+((s[3]-s[1])**2))**0.5
    r=10+r/2
    return p1,p2,r

def get_y_from_gcp():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    image_id = [*range(1,1197)]
    xml=[]

    for i in image_id:
        blob = bucket.get_blob(f'{BUCKET_TRAIN_y_PATH}/{i}.xml')
        y_gcp = blob.download_as_bytes()
        xml.append(y_gcp)
    return xml


def get_xml():
    y_list=[]
    xml = get_y_from_gcp()
    for i in xml:
        root = ET.fromstring(i)
        sample_annotations = []
        y_train=[]
        for neighbor in root.iter('bndbox'):
            xmin = float(neighbor.find('xmin').text)
            ymin = float(neighbor.find('ymin').text)
            xmax = float(neighbor.find('xmax').text)
            ymax = float(neighbor.find('ymax').text)
            sample_annotations.append([xmin, ymin, xmax, ymax])
            p1,p2,r=convert_box_to_circle([xmin, ymin, xmax, ymax])
            y_train.append([p1,p2,r])

        y=np.array(y_train)
        y=y.flatten()
        shape=y.shape[0]
        if shape<30:
            pad_num=30-shape
            y=np.pad(y,(0,pad_num))
        elif shape>30:
            y=y[:30]
        else:
            pass
        y_list.append(y)
    y=np.array(y_list)

    return y




if __name__ == '__main__':
    res = get_X_from_gcp()
    X = X_to_tensor(res)
    y = get_xml()
=======
from os.path import dirname, abspath, join
from dotenv import load_dotenv
import os



env_path = join(dirname(abspath(__file__)),'.env') # ../.env
load_dotenv(env_path)


def get_X_from_gcp():
    storage_client = storage.Client.from_service_account_json(os.getenv("gcp_json_path"))
    bucket = storage_client.bucket(BUCKET_NAME)

    num = [*range(1,590)]

    for i in num:
        blob = bucket.blob(f'{BUCKET_TRAIN_X_PATH}/{i}.jpg')
        img = blob.download_to_filename(f'/XPerts/data/{i}.jpg')


def X_to_tensor():
    num = [*range(1,590)]
    image = []

    for i in num:
        img = tf.keras.utils.load_img(f"./XPerts/data/{i}.jpg")
        image.append(tf.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0))
    X = tf.concat(image, 0)
    print(type(X))


def get_y_from_gcp():
    storage_client = storage.Client.from_service_account_json(os.getenv("gcp_json_path"))
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(BUCKET_TRAIN_y_PATH)
    blob.download_to_filename('/XPerts/data/data.csv')
    # data=data.drop(columns='Unnamed: 2')
    # data['Nu_Teeth']=data['Nu_Teeth'].replace(np.nan,'-1')
    # data['Nu_Teeth']=data['Nu_Teeth'].replace('B','-1')
    # y = data['Nu_Teeth']
    # y_cat = tf.keras.utilsto_categorical(y, num_classes=44)
    # y_cat = tf.convert_to_tensor(y_cat)
    # return y_cat

if __name__ == '__main__':
    # X = X_to_tensor()
    print('hoooi')
    y_cat = get_y_from_gcp()
>>>>>>> master
