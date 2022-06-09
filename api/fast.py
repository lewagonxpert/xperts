from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import io
import numpy as np
from keras.models import load_model
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, Form, Request
import tensorflow as tf
<<<<<<< HEAD
import base64






=======
from XPerts.params import BUCKET_NAME
import h5py
>>>>>>> c17040fa0559496913ee8d66b734be66aaf73382

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


<<<<<<< HEAD


@app.post("/image")
async def check_image(request: Request):
    request_body = await request.json()
    image_b64_utf8 = request_body["image"]
    return image_b64_utf8


@app.post("/preproc")
async def preproc(image_b64_utf8):
    size=(512,512)
    img_bytes = base64.b64decode(image_b64_utf8.encode('utf8'))

    # # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))
    return img



@app.get("/predict")
def predict():
    image_b64_utf8 = check_image()
    my_image = preproc(image_b64_utf8)
    image =[]
    img = Image.open(my_image)
    image.append(np.expand_dims(np.asarray(img),axis=0))

    # storage_client = storage.Client()
    # local_model_name = 'model.h5'
    # storage_location = f'model/xperts/v1/{local_model_name}'
    # blob = storage_client.blob(storage_location)
    # model_gcs = blob.download_as_bytes('model.h5')
    # model = load_model(io.BytesIO(model_gcs))


    model = load_model('model.h5')

    X_p=tf.concat(image, 0)
    y_p=model.predict(X_p)
    y_p=y_p.reshape(5,3)


    return {'y_p':y_p}





# if __name__ == '__main__':
#     predict()
=======
@app.post("/image/")
async def image_upload(file: UploadFile):
    # Read file to get bytes
    data = await file.read()
    # Return the result of run_model as api response
    # run_model return a numpy array, so convert to list, fastapi will convert to json for us
    return run_model(data).tolist()


def run_model(bytes):
    # Prepare image for processing
    image = Image.open(io.BytesIO(bytes))
    image_array = [np.expand_dims(np.asarray(image),axis=0)]
    X_p=tf.concat(image_array, 0)
    # Load our model
    client = storage.Client().bucket(BUCKET_NAME)
    local_model_name = 'model.h5'
    storage_location = f'model/xperts/v1/{local_model_name}'
    blob = client.blob(storage_location)
    model_gcs = blob.download_as_bytes()
    f = io.BytesIO(model_gcs)
    h = h5py.File(f,'r')
    model = load_model(h)
    # Get results
    y_p=model.predict(X_p)
    y_p=y_p.reshape(5,3)
    # Return results
    return y_p
>>>>>>> c17040fa0559496913ee8d66b734be66aaf73382
