from flask import Flask, render_template, jsonify, send_from_directory, request
import pickle
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
from flask_cors import CORS, cross_origin
import gc
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
PREDICT = 1

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(190, 250, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.load_weights('model_weights_81.54%.h5')

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')



@app.route("/images/<path:image_name>")
@cross_origin()
def get_image(image_name):
    image_folder = 'images'
    return send_from_directory(image_folder, image_name)

@app.route('/send', methods=['POST'])
@cross_origin()
def receive_image():
    global PREDICT
    image_bytes = request.data
    frame = pickle.loads(image_bytes)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_to_tensor = tf.convert_to_tensor(gray)
    img_to_tensor = tf.expand_dims(img_to_tensor, axis=-1)
    input_data = np.expand_dims(img_to_tensor, axis=0)
    
    predictions = model.predict(input_data, verbose=0)
    K.clear_session()
    PREDICT = int(np.argmax(predictions))
    gc.collect()

    return 'Image received successfully'

@app.route('/random_image')
@cross_origin()
def random_image():
    return jsonify({'image_path': '/images/' + str(PREDICT) + '.png'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
