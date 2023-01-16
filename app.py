from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import subprocess
from werkzeug.utils import secure_filename
import random
import image_processing
import requests
import os
import sched, time

flask = Flask(__name__)

UPLOAD_FOLDER = '/home/ubuntu/automatic-image-processing/uploads'
PROCESSED_FOLDER = '/home/ubuntu/automatic-image-processing/processed'

task_training = None
gender = None
instance_name = None
@flask.route('/submit', methods=['POST'])
def submit():
    global gender, instance_name
    if request.method == 'POST':

        if not is_training_running():
            training_subject = 'Character'
            subject_type = 'person'
            instance_name = ''.join(random.choice('abcdefghijklmnopqrtsvwyz') for i in range(10))
            class_dir = 'person_ddim'
            training_steps = 100
            seed = random.randint(7, 1000000)
            gender = request.form['gender']

            if 'images' not in request.files:
                return jsonify(message='No file uploaded', category="error", status=500)
            file = request.files['images']
            if file.filename == '':
                return jsonify(message='No file uploaded', category="error", status=500)

            filename = secure_filename(file.filename)
            try:
                file.save(os.path.join(UPLOAD_FOLDER, filename))
            except:
                return jsonify(message='The zip file could not be saved', category="error", status=500)

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
                        return jsonify(message='Zip contains other files than, png, jpg or jpeg', category="error", status=500)

            except Exception as e:
                return jsonify(message='An error occurred processing the zip file', category="error", status=500)

            subprocess.run(["rm", "-rf", UPLOAD_FOLDER + '/' + filename])

            subprocess.getoutput("mkdir -p " + PROCESSED_FOLDER + '/' + instance_name)

            for i, file in enumerate(os.listdir(UPLOAD_FOLDER + '/' + instance_name)):
                file_path = UPLOAD_FOLDER + '/' + instance_name + '/' + file
                extension = file.split('.')[-1]
                image_processing.test_image(file_path, PROCESSED_FOLDER + '/' + instance_name + '/' + str(i) + '.' + extension)

            zip_file = PROCESSED_FOLDER + '/' + instance_name + '/' + instance_name + '.zip'
            zip_files = PROCESSED_FOLDER + '/' + instance_name + '/*'
            subprocess.getoutput("zip -j {ZIP_FILE} {ZIP_FILES}".format(ZIP_FILE=zip_file , ZIP_FILES=zip_files))

            fileobj = open('test.zip', 'rb')
            payload = {'subject': training_subject, 'subject_type': subject_type, 'instance_name': instance_name, 'class_dir': class_dir, 'training_steps': training_steps, 'seed': seed}
            r = requests.post('http://localhost:3000/', data=payload, files={"images": (zip_file, fileobj)})

            if r.status_code != 200:
                return jsonify(message=r.text, category="error", status=500)

            task_training = sched.scheduler(time.time, time.sleep)
            task_training.enter(300, 1, check_if_training, (task_training,))
            task_training.run()


        return jsonify(message='Training started', category="success", status=200)

    return jsonify(message='Did not receive a POST request', category="error", status=500)


@flask.route('/status', methods=['GET'])
def get():
    return jsonify({'status': is_training_running()})


def is_training_running():
    output = subprocess.getoutput("ps -ef | grep accelerate | grep -v grep")
    return len(output) > 0


def check_if_training(runnable_task):
    if is_training_running():
        runnable_task.enter(300, 1, check_if_training, (runnable_task,))
    else:
        runnable_task.enter(900, 1, sd_ready, (runnable_task,))


def sd_ready(runnable_task):
    global gender, instance_name
    file_prompt = 'sd-' + gender + '-prompts.json'
    new_file_prompt = PROCESSED_FOLDER + '/' + instance_name + '/prompts.json'
    subprocess.getoutput("cp " + file_prompt + " " + new_file_prompt)
    subprocess.getoutput('sed -i "s/<subject>/' + instance_name + '/g" ' + new_file_prompt)
    fileobj = open(new_file_prompt, 'rb')

    r = requests.post('http://localhost:3000/txt2img', files={"prompts": ('prompts.json', fileobj)})


if __name__ == '__main__':
    flask.run(debug=True, port=6000)
