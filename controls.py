import cv2
from hand_detector import HandDetector
from pynput import mouse, keyboard

trKeyboard = keyboard.Controller()
trMouse = mouse.Controller()
detector = HandDetector(max_hands=1, track_con=0.85)
xp, yp = 0, 0
track_mode = False
command_counter = 0


def print_recognition_text(confidence, gesture, img):
    cv2.putText(img, 'confidence: ' + str(confidence), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, 1)
    cv2.putText(img, 'gesture: ' + gesture, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, 1)


def do_control(index):
    global track_mode, command_counter

    command_counter = (command_counter + 1) % 2

    if command_counter == 1:
        if index == 0:
            # trKeyboard.press(keyboard.Key.alt)
            # trKeyboard.press(keyboard.Key.tab)
            # trKeyboard.release(keyboard.Key.alt)
            # trKeyboard.release(keyboard.Key.tab)
            print("Command 1")
        elif index == 1:
            print("Command 2")


#
# Ignore this function
# ! Not used because there are problems with
#   hand tracking and gesture recognition concurrently
#
def track_mouse(frame):
    global xp, yp

    frame = detector.find(frame, draw=False)
    lmList = detector.find_points(frame)

    if len(lmList) != 0:
        cx, cy = lmList[8][1], lmList[8][2]

        if xp == 0 and yp == 0:
            xp, xy = cx, cy

        x, y = (xp - cx) * 4, (cy - yp) * 3.5
        trMouse.move(x, y)
        xp, yp = cx, cy

