import cv2
import tensorflow as tf
import controls
from gesture_detector import GestureDetector, config

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":
    recognizer = GestureDetector()
    recognizer.get_classes()

    confidence = 0.0
    gesture = ""
    gesture_index = 2

    xp, yp = 0, 0

    # capture video from USB web-camera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        result = recognizer.find_gesture(frame)
        if result is not None:
            (confidence, gesture, gesture_index) = result
            print(gesture)
            controls.do_control(gesture_index)

        controls.print_recognition_text(confidence, gesture, frame)

        if not config.get("quietMode"):
            cv2.imshow("Original", frame)

        # use ESC key to close the program
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
