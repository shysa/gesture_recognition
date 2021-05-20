import os
import cv2
import tensorflow as tf
import controls
from gesture_detector import GestureDetector, config
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
            first_x, first_y = lmlist[8][1], lmlist[8][2]
            sec_x, sec_y = lmlist[12][1], lmlist[12][2]
            third_x, third_y = lmlist[16][1], lmlist[16][2]
            fourth_x, fourth_y = lmlist[20][1], lmlist[20][2]

            flag1 = False
            flag2 = False
            flag3 = False
            flag4 = False

            if abs(first_x - lmlist[5][1]) <= 30 and abs(first_y - lmlist[5][2]) <= 30:
                flag1 = True
            if abs(sec_x - lmlist[9][1]) <= 30 and abs(sec_y - lmlist[9][2]) <= 30:
                flag2 = True
            if abs(third_x - lmlist[13][1]) <= 30 and abs(third_y - lmlist[13][2]) <= 30:
                flag3 = True
            if abs(fourth_x - lmlist[17][1]) <= 30 and abs(fourth_y - lmlist[17][2]) <= 30:
                flag4 = True

            if flag1 and flag2 and flag3 and flag4:
                mouse_mode = True
                skipped_frames = 0
                recognizer.clean_frames()

        else:
            mouse_mode = False
            skipped_frames += 1

        if mouse_mode:
            # get some coords for CLICK checking
            thumb_x, thumb_y = lmlist[4][1], lmlist[4][2]
            click_target_x, click_target_y = lmlist[5][1], lmlist[5][2]

            click_x = abs(thumb_x - click_target_x)
            click_y = abs(thumb_y - click_target_y)

            # and do click
            if click_x <= 30 and click_y <= 30:
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
            if skipped_frames >= 20:
                result = recognizer.find_gesture(frame)
                if result is not None:
                    (confidence, gesture, gesture_index) = result
                    print(gesture)
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
