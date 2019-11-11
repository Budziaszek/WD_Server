from flask import Flask, request, send_file
import numpy as np
import cv2

from inference_new import predicateImage

app = Flask(__name__)
image = "global"
mask = "global"


@app.route('/')
def hello_world():
    return 'Hello world!'


@app.route('/predicateImage')
def inference():
    global image, mask

    imagePath = predicateImage(image, mask)
    return send_file(imagePath, mimetype='image/png')


@app.route('/image', methods=['POST'])
def getImage():
    global image
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    image = cv2.resize(image, (256, 512))


@app.route('/mask', methods=['POST'])
def setMask():
    global mask
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    mask = cv2.imdecode(nparr, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    mask = cv2.resize(mask, (256, 512))


if __name__ == '__main__':
    # start flask app
    app.run(host="0.0.0.0", port=5000)