import os
from google.cloud import storage
from keras.models import load_model
import h5py
from XPerts.data import get_X_from_gcp, get_y_from_gcp,X_to_tensor,y_to_tensor
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH

PATH_TO_LOCAL_MODEL = 'model.h5'

def get_model(path_to_h5):
    model = load_model(path_to_h5)
    return model

def download_model(bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)
    local_model_name = 'model.h5'
    storage_location = f'model/xperts/v1/{local_model_name}'
    blob = client.blob(storage_location)
    model_gcs = blob.download_to_filename('model.h5')
    model = load_model(model_gcs)
    if rm:
        os.remove('model.h5')
    return model

def evaluate(model, X_test, y_test):
    return model.evaluate(X_test, y_test)
