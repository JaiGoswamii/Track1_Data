import logging
import socketio
from flask import Flask
import eventlet
import eventlet.wsgi
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np  # Import numpy
from keras.losses import mean_squared_error

logging.basicConfig(level=logging.DEBUG)

# Create a Socket.IO server instance
sio = socketio.Server()

# Create the Flask app
app = Flask(__name__)
speed_limit = 10

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

# Handle telemetry data from the client (simulator)
@sio.on('telemetry')
def telemetry(sid, data):
    # Log incoming telemetry data
    print("Received telemetry:", data)

    # Process image from the simulator
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])

    # Predict steering angle from the model
    steering_angle = float(model.predict(image))

    # Adjust throttle based on the speed
    throttle = 1.0 - speed / speed_limit
    if throttle < 0.1:
        throttle = 0.1  # Set a minimum throttle to keep the car moving

    # Log the values for debugging
    print('Steering angle: {}, Throttle: {}, Speed: {}'.format(steering_angle, throttle, speed))

    # Send control commands
    send_control(steering_angle, throttle)

# Define the connect event handler
@sio.on('connect')
def connect(sid, environ):
    print('Client connected:', sid)
    send_control(0, 0)

# Send control commands to the simulator
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    model = load_model('model.h5', custom_objects={'mse': mean_squared_error})
    app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
