
#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

interpreter = tflite.Interpreter(model_path='model_2024_hairstyle_v2.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    img = download_image(url)
    img = prepare_image(img, (200, 200))
    x = np.array(img, dtype = 'float32')
    x = np.array([x])
    x = x * (1./ 255)

    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    print(preds)
    print(type(preds))

    return float(preds[0, 0])

def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    print(type(pred))
    print(type(float(pred)))
    result = {
        'prediction' : float(pred)
    }

    return result