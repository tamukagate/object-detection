import cv2
import numpy as np
import tensorflow as tf
import os
from flask import Flask, request, jsonify, render_template, session, redirect, abort, send_file

app = Flask(__name__)
app.secret_key = 'secret'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['FRAME_FOLDER'] = 'frames/'

@app.route('/')
def index():
    return render_template('index.html')

# Set the maximum allowed file size (in bytes)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Define the input size and preprocessing function for the Inception V3 model
INPUT_SIZE = (299, 299)
def preprocess_input(frame):
    frame = cv2.resize(frame, INPUT_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = tf.expand_dims(frame, axis=0)
    return frame

# Load the Inception V3 model
model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights='imagenet',
    input_shape=(299, 299, 3),
    pooling=None,
    classes=1000
)

def extract_frames(video_path, frame_rate):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Extract frames at the specified frame rate
        if frame_count % frame_rate == 0:
            frames.append(frame)
    cap.release()
    return frames

def detect_objects(frame, model):
    # Preprocess the frame
    frame = preprocess_input(frame)
    # Feed the frame into the model
    preds = model.predict(frame)
    # Decode the predictions
    top_preds = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=1)[0]
    # Return the predicted class label and probability as a dictionary
    return {'class': top_preds[0][1], 'probability': float(top_preds[0][2])}

@app.route('/upload', methods=['POST'])
def handle_upload():
    # Check if the request method is POST
    if request.method == 'POST':
        # Check if the request contains a file object with the key "video"
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded.'}), 400
        
        video_file = request.files['video']
        
        # Check the size of the video file
        if video_file.content_length > MAX_FILE_SIZE:
            return jsonify({'error': 'Video file is too large.'}), 400
        
        # Save the video file to a designated folder on the server
        video_path = 'uploads/' + video_file.filename
        video_file.save(video_path)
        
        # Extract frames from the video
        frames = extract_frames(video_path, frame_rate=30)
        
        # Detect objects in each frame
        results = []
        for i, frame in enumerate(frames):
            frame_results = detect_objects(frame, model)
            frame_dict = {'frame': i, 'class': frame_results['class'], 'probability': frame_results['probability']}
            results.append(frame_dict)
        
        # Store the results in a session variable and redirect to the results page
        session['results'] = results
        return redirect('/results')
    
@app.route('/results')
def results():
    # Get the search query from the URL parameters
    search_query = request.args.get('q', '')
    # Get the results from the session variable
    results = session.get('results', [])
    # Filter the results to show only frames where the object was detected
    filtered_results = [r for r in results if search_query.lower() in r['class'].lower()]
    # Render
    return render_template('results.html', results=filtered_results, search_query=search_query)

@app.route('/frame/<int:frame_index>')
def serve_frame(frame_index):
    video_file = session.get('video_file')
    if not video_file:
        abort(404)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        abort(404)
    frame_path = os.path.join(app.config['FRAME_FOLDER'], 'frame_{}.jpg'.format(frame_index))
    cv2.imwrite(frame_path, frame)
    return send_file(frame_path)

if __name__ == '__main__':
    app.run(debug=True)
