import joblib
import os
from google.cloud import storage

from XPerts.data import get_X_from_gcp, get_y_from_gcp,X_to_tensor,y_to_tensor
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH



def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline

def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)
    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model


def evaluate_model(y, y_pred):
    return res
