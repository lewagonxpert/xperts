import pandas as pd
import numpy as np
from google.cloud import storage
import tensorflow as tf
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH
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
        if shape<15:
            pad_num=15-shape
            y=np.pad(y,(0,pad_num))
        elif shape>15:
            y=y[:15]
        else:
            pass
        y_list.append(y)
    y=np.array(y_list)

    return y




if __name__ == '__main__':
    res = get_X_from_gcp()
    X = X_to_tensor(res)
    y = get_xml()
