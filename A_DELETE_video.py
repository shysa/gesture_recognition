from collections import deque
import numpy as np
import cv2
import time
from keras.models import load_model
import tensorflow as tf
import controls

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# C O N F I G
model_path = "./saved_model/3DCNN+3LSTM_64_6_jester"
quietMode = False

# D E F A U L T
img_rows, img_cols = 64, 64
font = cv2.FONT_HERSHEY_SIMPLEX
width, height = 640, 480

Q = deque(maxlen=4)

if __name__ == "__main__":
    # load saved model from config path
    model = load_model(model_path)

    # capture video from USB web-camera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # set frames of the video stream width and height
    cap.set(3, width)
    cap.set(4, height)

    framecount = 0
    fps = ""
    start = time.time()
    frames = []
    num = [2]
    max = 1
    real_index = 5
    instruction = "No gesture"
    pre = 0

    # need to config (separate to csv)
    classes = ["Swipe Left", "Swipe Right", "No gesture"]
    num_classes = len(classes)

    while True:
        (ret, frame) = cap.read()
        frame = cv2.flip(frame, 3)
        frame = cv2.resize(frame, (width, height))
        
        framecount = framecount + 1
        end = time.time()
        timediff = (end - start)
        if timediff >= 1:
            fps = 'FPS:%s' % framecount
            start = time.time()
            framecount = 0

        # write info about fps on the video
        # params:
        # frame to output
        # text to output
        # coords of the bottom-left corner of the text
        # font type
        # font scale
        # color
        # thickness
        # line type
        cv2.putText(frame, fps, (10, 20), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        X_tr = []

        # image decimation with resampling using pixel area relation
        image = cv2.resize(frame, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(gray)
        input = np.array(frames)
        
        if input.shape[0] == 16:
            frames = []
            # print(input.shape)
            X_tr.append(input)
            X_train = np.array(X_tr)
            # print(X_train.shape)
            train_set = np.zeros((1, 16, img_cols, img_rows, 3))
            train_set[0][:][:][:][:] = X_train[0, :, :, :, :]
            train_set = train_set.astype('float32')
            train_set -= np.mean(train_set)
            train_set /= np.max(train_set)
            result = model.predict(train_set)
            Q.append(result)
            num = np.argmax(np.array(Q).mean(axis=0))
            instruction = classes[int(num)]
            print(instruction, classes[int(np.argmax(result))])
            input = []
            pre = int(num)

        cv2.putText(frame, instruction, (450, 50), font, 0.7, (0, 255, 0), 2, 1)

        if not quietMode:
            cv2.imshow('Original', frame)
        key = cv2.waitKey(1) & 0xFF
        # use Esc key to close the program
        if key == 27:
            break
        elif key == ord('q'):
            quietMode = not quietMode
            print("Quiet Mode - {}".format(quietMode))

cap.release()
cv2.destroyAllWindows()    

