from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import numpy as np
from google.cloud import storage
from keras.models import load_model
import h5py
from XPerts.params import BUCKET_NAME
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, Form
import tensorflow as tf


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



@app.post('/image')
async def _image_upload(my_image: bytes = File(...)):
    return my_image



@app.get("/predict")
def predict():
    my_image = _image_upload()
    image =[]
    img = Image.open(io.BytesIO(my_image))
    image.append(np.expand_dims(np.asarray(img),axis=0))

    storage_client = storage.Client()
    local_model_name = 'model.h5'
    storage_location = f'model/xperts/v1/{local_model_name}'
    blob = storage_client.blob(storage_location)
    model_gcs = blob.download_as_bytes('model.h5')
    model = load_model(io.BytesIO(model_gcs))
    print(type(model))

    X_p=tf.concat(image, 0)
    y_p=model.predict(X_p)
    y_p=y_p.reshape(10,3)



    sample_image_annotated = img.copy()
    img_bbox = ImageDraw.Draw(sample_image_annotated)
    i=0
    for bbox in y_p:
    #     print(bbox)
        if bbox[0]>1 and bbox[1]>1 and bbox[2]>10:
            i=i+1
            print(f'I Find {i}th cavity')
            img_bbox.regular_polygon(bounding_circle=(int(bbox[0]),int(bbox[1]),int(bbox[2])),n_sides=500, outline="red",)
    return {sample_image_annotated :'sample_image_annotated'}



if __name__ == '__main__':
    predict()
