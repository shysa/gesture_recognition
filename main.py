import os
import cv2
import tensorflow as tf
import controls
from gesture_detector import GestureDetector, config, find_static_gesture, find_click
from hand_detector import HandDetector

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if __name__ == "__main__":
    recognizer = GestureDetector()
    recognizer.get_classes()

    detector = HandDetector(max_hands=1, track_con=0.85)

    confidence = [0.0]
    gesture = "None"
    gesture_index = 0

    mouse_mode = False
    k = [1920/640, 1080/480]

    # capture video from USB web-camera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Источник", cv2.WINDOW_AUTOSIZE)

    # for pause after mouse mode
    skipped_frames = 0

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # detect hand
        img = detector.find(frame, draw=False)

        # find coord for joints for CLICK gesture tracking
        lmlist = detector.find_points(img)

        if len(lmlist) != 0:
            # check gesture for mouse mode ON
            mouse_on = find_static_gesture(lmlist)

            if mouse_on:
                mouse_mode = True
                skipped_frames = 0
                recognizer.clean_frames()

        else:
            # some pause for recognizer after mouse mode off
            mouse_mode = False
            skipped_frames += 1

        if mouse_mode:
            # check click gesture
            (get_click, click_target_x, click_target_y, thumb_x, thumb_y) = find_click(lmlist)

            if get_click:
                controls.do_control(10)
                cv2.circle(img, (click_target_x, click_target_y), 10, (91, 94, 255), cv2.FILLED)
                cv2.circle(img, (thumb_x, thumb_y), 10, (91, 94, 255), cv2.FILLED)

            # draw cursor point on index finger and for click
            index_x, index_y = lmlist[8][1], lmlist[8][2]
            cv2.circle(img, (index_x, index_y), 5, (91, 94, 255), cv2.FILLED)

            cv2.circle(img, (click_target_x, click_target_y), 10, (91, 94, 255), thickness=1)
            cv2.circle(img, (thumb_x, thumb_y), 10, (91, 94, 255), thickness=1)
            cv2.line(img, (click_target_x, click_target_y), (thumb_x, thumb_y), (91, 94, 255), thickness=1)

            controls.move_mouse(lmlist, k)
            controls.print_text("Mouse mode enabled", frame)

        else:
            # start recognize gestures after mouse mode off and 25 frames
            if skipped_frames >= 25:
                result = recognizer.find_gesture(cv2.flip(frame, 1))

                if result is not None:
                    (confidence, gesture, gesture_index) = result
                    controls.do_control(gesture_index)

                controls.print_recognition_text(confidence, gesture, frame)
            else:
                controls.print_text("Recognition mode enabling, wait", frame)

        if not config.get("quietMode"):
            cv2.imshow("Источник", frame)

        # use ESC key to close the program
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
