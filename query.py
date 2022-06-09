import requests
import os
import numpy as np
import base64
import json


# Local URLs and endpoints.
URL_base    = "http://127.0.0.1:8000/"



print("--------------------------------------------------")
print("----------------Query local docker----------------")
print("--------------------------------------------------")

image_path='./601.jpg'
print(os.path.exists(image_path))

print('\n\n')
# Get response from local base.
print('Response from base local  URL:')
response = requests.get(URL_base)
print(response)
print( response.json() )
print('\n\n')


with open(image_path, "rb") as f:
    im_bytes = f.read()
im_b64 = base64.b64encode(im_bytes).decode("utf8")
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
payload = json.dumps({"image": im_b64})

URL_image=URL_base+'image'

print('\n\nResponse from local poison  endpoint')
response = requests.post(URL_image, headers=headers,data=payload)
print(response)
print(response.json())
print('\n\n')


# Get response from local image
response = requests.post(URL_image, data=payload, headers=headers)
print("--------------------------------------------------")
print(response)
print(response.json())
print("---------------------------------------------------")



# print(im_b64)
URL_predict=URL_base+'predict'

response = requests.post(URL_image, data=payload, headers=headers)
print("--------------------------------------------------")
print(response)
print(response.json())
print("---------------------------------------------------")
