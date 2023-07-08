import cv2
import logging
import pickle
import requests
from pygrabber.dshow_graph import FilterGraph
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(filename='errors.log', level=logging.ERROR)
URL = os.getenv('URL')
TIME_TO_CONNECT = float(os.getenv('TIME_TO_CONNECT'))

def sendImage(session, frame):
    image_bytes = pickle.dumps(frame)
    try:
        session.post(URL, data=image_bytes, timeout = TIME_TO_CONNECT)
    except requests.exceptions.Timeout:
        logging.error("Sending image: %s", 'Connection timeout')
        print('ERROR: Connection timeout')
        pass


def mainLoop(getCamera):
    try:
        session = requests.Session()
        graph_cam = cv2.VideoCapture(getCamera)
        print('Началась передача изображений...')
        while True:
            ret, frame = graph_cam.read()
            if not ret:
                logging.error("Reading the stream from the camera: %s", 'Failed to read image from camera')
                break
            resized_img = cv2.resize(frame, (250, 190))
            sendImage(session, resized_img)
            
    except ValueError as e:
        logging.error("Main image transmission loop: %s", str(e))
    finally:
        session.close()

def start():
    graph = FilterGraph()

    print('Выберите вебкамеру')
    cameras = graph.get_input_devices()
    if len(cameras) == 0:
        logging.error("Getting a webcam: %s", 'No webcams found')
        exit(1)
        
    for cameraIndex in range(0, len(cameras)):
        print(f'{cameraIndex}: {cameras[cameraIndex]}')

    getCamera = int(input("Индекс выбранной камеры: "))

    if getCamera not in range(0, len(cameras)): 
        logging.error("Getting webcams: %s", f'''
                    Invalid index selected
                    Contains: {cameras}
                    received: {getCamera}
                    ''')
        exit(1)
    return getCamera

if __name__ == '__main__':
    getCamera = start()
    mainLoop(getCamera)
