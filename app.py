from flask import Flask, request, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired, FileSize
import cv2
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SECRET_KEY'] = 'secret'

model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')

class VideoForm(FlaskForm):
    video = FileField('Video', validators=[FileRequired(), FileAllowed(['mp4', 'mkv'], 'Only mp4 and mkv videos are allowed.'), FileSize(max_size=100*1024*1024, message='Video size must be less than 100MB.')])

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    form = VideoForm()

    if request.method == 'POST':
        if form.validate_on_submit():
            video = form.video.data
            video_path = os.path.join('uploads', video.filename)
            max_file_size = 100 * 1024 * 1024 # 10MB maximum file size
            if video.content_length > max_file_size:
                form.video.errors.append(f'File size exceeds {max_file_size/1024/1024}MB.')
            else:
                video.save(video_path)
                frame_dir = os.path.join('frames', video.filename.split('.')[0])
                os.makedirs(frame_dir, exist_ok=True)
                num_frames = video_to_frames(video_path, frame_dir)
                objects = detect_objects(frame_dir)
                return render_template('result.html', objects=objects)
    return render_template('upload.html', form=form)

def video_to_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(os.path.join(output_path, f'frame{count}.jpg'), frame)
        count += 1
    cap.release()
    return count

def detect_objects(frame_dir):
    images = []
    for file_name in os.listdir(frame_dir):
        if file_name.endswith('.jpg'):
            img_path = os.path.join(frame_dir, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            images.append(img)
    images = np.vstack(images)
    preds = model.predict(images)
    results = tf.keras.applications.inception_v3.decode_predictions(preds, top=3)
    objects = []
    for result in results:
        for r in result:
            objects.append(r[1])
    return objects
        
if __name__ == '__main__':
    app.run(debug=True)