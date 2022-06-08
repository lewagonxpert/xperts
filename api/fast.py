from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
from keras.models import load_model
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, Form, Request
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
    model = load_model('model_xperts_v2_model.h5')
    # Get results
    y_p=model.predict(X_p)
    y_p=y_p.reshape(5,3)
    # Return results
    return y_p
