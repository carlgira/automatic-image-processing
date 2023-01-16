from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import subprocess
from werkzeug.utils import secure_filename
import random
import image_processing
import os

flask = Flask(__name__)

UPLOAD_FOLDER = '/home/ubuntu/automatic-image-processing/uploads'
PROCESSED_FOLDER = '/home/ubuntu/automatic-image-processing/processed'

@flask.route('/submit', methods=['POST'])
def home():
    if request.method == 'POST':

        if not is_training_running():
            training_subject = 'Character'
            subject_type = 'person'
            instance_name = ''.join(random.choice('abcdefghijklmnopqrtsvwyz') for i in range(10))
            class_dir = 'person_ddim'
            training_steps = 1600
            seed = random.randint(7, 1000000)

            if 'images' not in request.files:
                return
            file = request.files['images']
            if file.filename == '':
                return jsonify({'code': 'error', 'message': 'No file uploaded'})

            filename = secure_filename(file.filename)
            try:
                file.save(os.path.join(UPLOAD_FOLDER, filename))
            except:
                return jsonify({'code': 'error', 'message': 'The zip file could not be saved'})

            try:
                subprocess.run(["unzip", "-j" , UPLOAD_FOLDER + '/' + filename, '-d' , UPLOAD_FOLDER + '/' + instance_name], check=True)

                for i, file in enumerate(os.listdir(UPLOAD_FOLDER + '/' + instance_name)):
                    file_path = UPLOAD_FOLDER + '/' + instance_name + '/' + file
                    if os.path.isdir(file_path):
                        subprocess.run(["rm", "-rf", file_path])
                        continue

                    # check if hidden file
                    if file[0] == '.':
                        os.remove(file_path)
                        continue

                    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        return jsonify({'code': 'error', 'message': 'Zip contains other files than, png, jpg or jpeg'})

            except Exception as e:
                return jsonify({'code': 'error', 'message': 'An error occurred processing the zip file'})

            subprocess.run(["rm", "-rf", UPLOAD_FOLDER + '/' + filename])

            subprocess.getoutput("mkdir -p " + UPLOAD_FOLDER + '/' + instance_name)

            for i, file in enumerate(os.listdir(UPLOAD_FOLDER + '/' + instance_name)):
                file_path = UPLOAD_FOLDER + '/' + instance_name + '/' + file
                extension = file.split('.')[-1]
                image_processing.test_image(file_path, PROCESSED_FOLDER + '/' + instance_name + '/' + str(i) + '.' + extension)

            zip_file = PROCESSED_FOLDER + '/' + instance_name + '/' + instance_name + '.zip'
            zip_files = PROCESSED_FOLDER + '/' + instance_name + '/*'
            subprocess.getoutput("zip -j {ZIP_FILE} {ZIP_FILES}".format(ZIP_FILE=zip_file , ZIP_FILES=zip_files))

        return jsonify({'code': 'ok', 'message': 'Training started'})


@flask.route('/status', methods=['GET'])
def get():
    return jsonify({'status': is_training_running()})


def is_training_running():
    output = subprocess.getoutput("ps -ef | grep accelerate | grep -v grep")
    return len(output) > 0


if __name__ == '__main__':
    flask.run(debug=True, port=6000)
