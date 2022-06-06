import pandas as pd
import numpy as np
from google.cloud import storage
import tensorflow as tf
from XPerts.params import BUCKET_NAME, BUCKET_TRAIN_X_PATH,BUCKET_TRAIN_y_PATH
from PIL import Image
import io
from io import StringIO
from os import listdir
from os.path import isfile, join
import matplotlib.patches as ptc
import xml.etree.ElementTree as ET


def get_X_from_gcp():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)

    image_id = [582,81,350,128,440,331,323,390,448,353,543,577,501,4,213,473,269,398,206,17,322,588,147,150,241,546,417,117,214,243,112,471,93,552,52,401,405,278,15,89,79,100,434,229,341,437,497,314,313,250,279,144,387,13,332,273,207,522,271,33,205,469,525,272,34,155,259,216,277,45,73,43,571,516,310,292,579,451,480,168,557,550,394,575,485,164,427,414,408,202,478,105,101,215,76,456,447,505,359,366,130,461,586,233,474,255,519,284,324,560,109,247,257,567,83,198,70,171,38,234,515,137,217,29,60,289,99,282,224,25,496,312,151,194,346,218,193,549,48,561,385,535,433,429,78,406,113,126,219,199,301,123,196,249,59,97,121,381,68,316,228,380,211,481,439,531,337,166,264,565,71,39,111,465,370,477,263,410,330,460,555,507,16,163,375,369,542,288,176,517,84,145,445,325,203,391,18,374,493,64,317,116,2,367,334,88,382,527,37,364,266,361,77,488,328,46,349,173,563,185,297,578,333,54,262,181,520]

    res = []

    for i in image_id:
        try:
            blob = bucket.get_blob(f'{BUCKET_TRAIN_X_PATH}/{i}.jpg')
            test = blob.download_as_bytes()
            res.append(test)
        except Exception as e: print(e)
    return res


def X_to_tensor(res):
    image = []
    for i in res:
        img = Image.open(io.BytesIO(i))
        image.append(np.expand_dims(np.asarray(img),axis=0))
    X = tf.concat(image, 0)
    return X


def convert_box_to_circle(s):
    p1=int((s[0]+s[2])/2)
    p2=int((s[1]+s[3])/2)
    r=(((s[2]-s[0])**2)+((s[3]-s[1])**2))**0.5
    r=10+r/2
    return p1,p2,r

def get_y_from_gcp():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    image_id = [582,81,350,128,440,331,323,390,448,353,543,577,501,4,213,473,269,398,206,17,322,588,147,150,241,546,417,117,214,243,112,471,93,552,52,401,405,278,15,89,79,100,434,229,341,437,497,314,313,250,279,144,387,13,332,273,207,522,271,33,205,469,525,272,34,155,259,216,277,45,73,43,571,516,310,292,579,451,480,168,557,550,394,575,485,164,427,414,408,202,478,105,101,215,76,456,447,505,359,366,130,461,586,233,474,255,519,284,324,560,109,247,257,567,83,198,70,171,38,234,515,137,217,29,60,289,99,282,224,25,496,312,151,194,346,218,193,549,48,561,385,535,433,429,78,406,113,126,219,199,301,123,196,249,59,97,121,381,68,316,228,380,211,481,439,531,337,166,264,565,71,39,111,465,370,477,263,410,330,460,555,507,16,163,375,369,542,288,176,517,84,145,445,325,203,391,18,374,493,64,317,116,2,367,334,88,382,527,37,364,266,361,77,488,328,46,349,173,563,185,297,578,333,54,262,181,520]

    xml=[]

    for i in image_id:
        blob = bucket.get_blob(f'{BUCKET_TRAIN_y_PATH}/{i}.xml')
        y_gcp = blob.download_as_string()
        xml.append(y_gcp)
    return xml


def get_xml(xml):

    tree = ET.parse(xml)
    root = tree.getroot()

    sample_annotations = []
    y_train=[]
    for neighbor in root.iter('bndbox'):
        xmin = float(neighbor.find('xmin').text)
        ymin = float(neighbor.find('ymin').text)
        xmax = float(neighbor.find('xmax').text)
        ymax = float(neighbor.find('ymax').text)
        sample_annotations.append([xmin, ymin, xmax, ymax])
        p1,p2,r=convert_box_to_circle([xmin, ymin, xmax, ymax])
        y_train.append([p1,p2,r])

    y=np.array(y_train)
    y=y.flatten()
    shape=y.shape[0]
    if shape<30:
        pad_num=30-shape
        y=np.pad(y,(0,pad_num))
    elif shape>30:
        y=y[:30]
    else:
        pass
    return y


def get_y():
    y=[]
    num = [582,81,350,128,440,331,323,390,448,353,543,577,501,4,213,473,269,398,206,17,322,588,147,150,241,546,417,117,214,243,112,471,93,552,52,401,405,278,15,89,79,100,434,229,341,437,497,314,313,250,279,144,387,13,332,273,207,522,271,33,205,469,525,272,34,155,259,216,277,45,73,43,571,516,310,292,579,451,480,168,557,550,394,575,485,164,427,414,408,202,478,105,101,215,76,456,447,505,359,366,130,461,586,233,474,255,519,284,324,560,109,247,257,567,83,198,70,171,38,234,515,137,217,29,60,289,99,282,224,25,496,312,151,194,346,218,193,549,48,561,385,535,433,429,78,406,113,126,219,199,301,123,196,249,59,97,121,381,68,316,228,380,211,481,439,531,337,166,264,565,71,39,111,465,370,477,263,410,330,460,555,507,16,163,375,369,542,288,176,517,84,145,445,325,203,391,18,374,493,64,317,116,2,367,334,88,382,527,37,364,266,361,77,488,328,46,349,173,563,185,297,578,333,54,262,181,520]
    for i in num:
        y.append(get_xml(i))
    y=np.array(y)
    return y



if __name__ == '__main__':
    res = get_X_from_gcp()
    X = X_to_tensor(res)
    xml = get_y_from_gcp()
    y = get_y()
