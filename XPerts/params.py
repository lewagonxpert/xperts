<<<<<<< HEAD
BUCKET_NAME = 'x-perts-907'
BUCKET_TRAIN_X_PATH = 'data/pascal/JPEGImages'
BUCKET_TRAIN_y_PATH = 'data/pascal/Annotations'

MODEL_NAME = 'X-Perts'
MODEL_VERSION = 'v1'
=======
### MLFLOW configuration - - - - - - - - - - - - - - - - - - -


### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

# PATH_TO_LOCAL_MODEL = 'model.joblib'

# AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"


### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'x-perts-907'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_X_PATH = 'data/X'
BUCKET_TRAIN_y_PATH = 'data/y/DHI_Count.csv'
##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'X-Perts'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -
>>>>>>> master
