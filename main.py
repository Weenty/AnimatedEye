import sys
import logging
logging.basicConfig(filename='app.log', level=logging.ERROR)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QComboBox, QDialog
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
import time
import gc
# массивы используются с задумкой на дальнейшее улучшение анимации (добавление фреймов для плавности)
stay_mid = [QImage('gitImg/'+'7.png'), 
            QImage('gitImg/'+'7.png')]
stay_right = [QImage('gitImg/'+'10.png'), 
              QImage('gitImg/'+'10.png')]
stay_left = [QImage('gitImg/'+'4.png'), 
             QImage('gitImg/'+'4.png')]

cap = cv2.VideoCapture(0)

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


class AnimationUpdater(QObject):
    updateAnimation = pyqtSignal(object)

    def __init__(self, camera_index, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.running = True

    def start(self):
        prev_class = 'mid'
        animation_images = stay_mid
        prev_animation_index = 0
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized_img = cv2.resize(gray, (250, 190))
                img_to_tensor = tf.convert_to_tensor(resized_img)
                img_to_tensor = tf.expand_dims(img_to_tensor, axis=-1)
                input_data = np.expand_dims(img_to_tensor, axis=0)

                predictions = model.predict(input_data, verbose=0)
                K.clear_session()
                predicted_class_index = np.argmax(predictions)
                predicted_class = ['left', 'right', 'mid'][predicted_class_index]

                
                if predicted_class == 'left':
                    animation_images = stay_left
                elif predicted_class == 'right':
                    animation_images = stay_right
                elif predicted_class == 'mid':
                    animation_images = stay_mid

                if prev_class != predicted_class:
                    prev_class = predicted_class
                    prev_animation_index = 0

                current_animation_image = animation_images[prev_animation_index]
                self.updateAnimation.emit(current_animation_image)
                time.sleep(0.15)
                gc.collect()
                prev_animation_index = (prev_animation_index + 1) % len(animation_images)
        except Exception as e:
            logging.error("Exception in AnimationUpdater: %s", str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Transparent Window")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.image = QImage("gitImg/1.png")
        self.background_label = QLabel(self)
        self.background_label.setPixmap(QPixmap.fromImage(self.image))
        self.layout.addWidget(self.background_label)
    
    def update_animation(self, image):
        self.background_label.setPixmap(QPixmap.fromImage(image))
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = event.globalPos() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPos()
            
    def start_animation(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)
        self.animation_updater = AnimationUpdater(camera_index)
        self.animation_updater.updateAnimation.connect(self.update_animation)
        self.animation_thread = QThread()
        self.animation_updater.moveToThread(self.animation_thread)
        self.animation_thread.started.connect(self.animation_updater.start)
        self.animation_thread.start()

        
    def cancel_animation(self):
        self.animation_updater.running = False
        self.cap.release()

# Изначально в этом окне можно было выбирать камеру, но 
# я её забил заглушкой (нулем), чтобы цеплялась камера "по-умолчанию"
class CameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Camera Selection')
        self.layout = QVBoxLayout(self)

        self.camera_combobox = QComboBox()
        self.layout.addWidget(self.camera_combobox)

        self.populate_camera_combobox()

    def populate_camera_combobox(self):
        self.camera_combobox.addItem("Default Camera")

    def selected_camera_index(self):
        return 0

    @staticmethod
    def get_camera_index(parent=None):
        dialog = CameraDialog(parent)
        camera_index = dialog.selected_camera_index()
        return camera_index, True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    camera_index, accepted = CameraDialog.get_camera_index(main_window)
    if accepted:
        main_window.start_animation(camera_index)
        sys.exit(app.exec_())
