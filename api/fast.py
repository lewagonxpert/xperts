from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
#from predict import download_model
import joblib

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


# @app.get("/predict")
# def predict(pickup_datetime, pickup_longitude,pickup_latitude,
#             dropoff_longitude, dropoff_latitude, passenger_count):

#     pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
#     eastern = pytz.timezone("US/Eastern")
#     localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
#     utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
#     formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
#     data = {
#         "key" : ["2013-07-06 17:18:00.000000119"],
#         "pickup_datetime": [formatted_pickup_datetime],
#         "pickup_longitude": [float(pickup_longitude)],
#         "pickup_latitude": [float(pickup_latitude)],
#         "dropoff_longitude": [float(dropoff_longitude)],
#         "dropoff_latitude": [float(dropoff_latitude)],
#         "passenger_count": [int(passenger_count)]
#     }
#     X_pred = pd.DataFrame(data)
#     model = joblib.load("model.joblib")
#     return model.predict(X_pred).tolist()[0]
