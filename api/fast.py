from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from google.cloud import storage
from keras.models import load_model
import h5py
from XPerts.params import BUCKET_NAME



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    local_model_name = 'model.h5'
    storage_location = f'model/xperts/v1/{local_model_name}'
    blob = storage_client.blob(storage_location)
    model_gcs = blob.download_to_filename('model.h5')
    model = load_model(model_gcs)
