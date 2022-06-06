import joblib
import os
from google.cloud import storage
from keras.models import load_model
import cv2

from XPerts.data import get_X_from_gcp, get_y_from_gcp,X_to_tensor,y_to_tensor
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH



def get_model(path_to_joblib):
    model = load_model(path_to_joblib)
    return model

def download_model(model_directory="model", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)
    MODEL_NAME = model
    storage_location = 'model/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.h5')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.h5')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.h5')
    if rm:
        os.remove('model.h5')
    return model
