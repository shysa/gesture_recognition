from collections import deque
import numpy as np
import cv2
from keras.models import load_model
import yaml

config = yaml.safe_load(open("./config.yaml"))

img_rows, img_cols = 64, 64
font = cv2.FONT_HERSHEY_SIMPLEX
width, height = 640, 480


class GestureDetector:
    def __init__(self, stab=deque(maxlen=3)):
        self.quiet_mode = config['quietMode']
        self.model_path = config['modelpath']
        self.model = load_model(self.model_path)
        self.csv_paths = config['csvpath']
        self.frames = []

        # define deque Q to stabilize predictions from recognizer
        self.Q = stab

    def get_classes(self):
        # TODO: parse csv
        self.classes = ["No gesture", "Sliding Two Fingers Down", "Sliding Two Fingers Up", "Swipe Left", "Swipe Right"]
        self.num_classes = len(self.classes)

    def find_gesture(self, frame, full_pred=False):
        img = cv2.resize(frame, (width, height))

        # image decimation with resampling using pixel area relation
        img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # form arrays of frames and input sequences for cnn
        self.frames.append(gray)
        input = np.array(self.frames)
        X_tr = []

        if input.shape[0] == 16:
            X_tr.append(input)
            X_train = np.array(X_tr)

            # preparing video sequence
            train_set = np.zeros((1, 16, img_cols, img_rows, 3))
            train_set[0][:][:][:][:] = X_train[0, :, :, :, :]
            train_set = train_set.astype('float32')
            train_set -= np.mean(train_set)
            train_set /= np.max(train_set)

            prediction = self.model.predict(train_set)

            # stabilize prediction
            self.Q.append(prediction)

            class_num = np.argmax(np.array(self.Q).mean(axis=0), axis=1)
            instruction = self.classes[int(class_num)]

            # clean arrays
            self.frames = []
            input = []

            if full_pred:
                return prediction[0], instruction, class_num
            else:
                return prediction[0][class_num], instruction, class_num

    def clean_frames(self):
        self.frames = []
        self.Q.clear()
