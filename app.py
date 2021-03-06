import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import numpy as np
import cv2
from werkzeug.utils import secure_filename

from inference_new import predicateImage

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def mainPage():
    if request.method == 'POST':
        return redirect(url_for('getImage'))
    else:
        return render_template('main_page.html', title='Home')


@app.route('/image', methods=['GET', 'POST'])
def getImage():
    if request.method == 'POST':
        if request.form['button'] == 'Cancel':
            print("Cancel")
            return redirect(url_for('mainPage'))
        else:
            if 'file_image' not in request.files or 'file_mask' not in request.files:
                print('No file')
                return redirect(request.url)
            file_image = request.files['file_image']
            file_mask = request.files['file_mask']
            if file_image.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file_image:
                filename_image = secure_filename(file_image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_image)
                file_image.save(image_path)
                file = open(image_path)
                file_image.stream.seek(0)
                # convert bfile to uint8
                nparr_image = np.fromfile(file, np.uint8)
                # decode image
                image = cv2.imdecode(nparr_image, cv2.IMREAD_REDUCED_GRAYSCALE_8)
                image = cv2.resize(image, (256, 512))

                filename_mask = secure_filename(file_mask.filename)
                mask_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_mask)
                file_mask.save(mask_path)
                file = open(mask_path)
                file_mask.stream.seek(0)
                # convert file to uint8
                nparr_mask = np.fromfile(file, np.uint8)
                # decode mask
                mask = cv2.imdecode(nparr_mask, cv2.IMREAD_REDUCED_GRAYSCALE_8)
                mask = cv2.resize(mask, (256, 512))

                filename_result = "result_" + filename_image
                path_result = os.path.join(app.config['UPLOAD_FOLDER'], filename_result)
                predicateImage(image, image_path, mask, mask_path, path_result, app.config['UPLOAD_FOLDER'])
                return redirect(url_for('displayResults', image_name=filename_image, mask_name=filename_mask,
                                        result_name=filename_result))
    else:
        return render_template('load_image.html', title='Home')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/show/<image_name>/<mask_name>/<result_name>', methods=['GET', 'POST'])
def displayResults(image_name, mask_name, result_name):
    if request.method == 'POST':
        if request.form['button'] == 'Return':
            return redirect(url_for('mainPage'))
    mimage = 'http://127.0.0.1:5000/uploads/' + image_name
    mmask = 'http://127.0.0.1:5000/uploads/' + mask_name
    mresult = 'http://127.0.0.1:5000/uploads/' + result_name
    return render_template('image_display.html', image=mimage, mask=mmask, result=mresult)


if __name__ == '__main__':
    # start flask app
    app.run(host="127.0.0.1", port=5000)
