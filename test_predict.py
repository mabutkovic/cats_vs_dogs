import requests
import base64
import json
import cv2
import numpy as np

classes = ["cat","dog"]
#image = r"12498.jpg" # dog
image = r"89.jpg" # cat
URL = "http://127.0.0.1:8501/v1/models/model/versions/1:predict"
headers = {"content-type": "application/json"}

headers = {"content-type": "application/json"}
image_content = cv2.imread(image,1).astype('uint8')
width = 150
height = 150
dim = (width, height)

resized = cv2.resize(image_content, dim, interpolation = cv2.INTER_AREA)
print('Original Dimensions : ',image_content.shape)
print('Original Dimensions : ',resized.shape)
body = {"instances": [resized.tolist()]}

data = json.dumps(body)
r = requests.post(URL, data=data, headers = headers)
predictions = json.loads(r.text)['predictions']
print(r.text)
print(classes[np.argmax(predictions[0])])
