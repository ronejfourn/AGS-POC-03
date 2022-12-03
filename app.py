from flask import Flask, request
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)

sess = ort.InferenceSession("mnist-8.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

def process_image(img):
    img = Image.open(img)
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    return img

def index_of_max(arr):
    m = 0
    for i in range(len(arr)):
        if (arr[i] > arr[m]):
            m = i
    return m

@app.route('/')
def nothing():
    return "Nothing to see here :)"

@app.route('/guess', methods=['POST'])
def guess():
    file = request.files['image']
    img = process_image(file)
    arr = (np.array(img) / 255).astype(np.float32)
    arr = np.reshape(arr, (1, 1, 28, 28))
    prediction = sess.run([label_name], {input_name: arr})[0]
    output = index_of_max(prediction[0])
    return str(output)

if __name__ == '__main__':
    app.run(port = 5000)
