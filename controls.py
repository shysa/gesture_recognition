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
        # Gesture: Swiping Left
        # Action: open prev tab
        if index == 3:
            trKeyboard.press(keyboard.Key.ctrl)
            trKeyboard.press(keyboard.Key.shift)
            trKeyboard.press(keyboard.Key.tab)
            trKeyboard.release(keyboard.Key.ctrl)
            trKeyboard.release(keyboard.Key.shift)
            trKeyboard.release(keyboard.Key.tab)

        # Gesture: Swiping Right
        # Action: open next tab
        elif index == 4:
            trKeyboard.press(keyboard.Key.ctrl)
            trKeyboard.press(keyboard.Key.tab)
            trKeyboard.release(keyboard.Key.ctrl)
            trKeyboard.release(keyboard.Key.tab)

        # Gesture: Sliding Two Fingers Up
        # Action: scroll down a frame
        elif index == 2:
            trKeyboard.press(keyboard.Key.space)
            trKeyboard.release(keyboard.Key.space)

        # Gesture: Sliding Two Fingers Down
        # Action: scroll up a frame
        elif index == 1:
            trKeyboard.press(keyboard.Key.shift)
            trKeyboard.press(keyboard.Key.space)
            trKeyboard.release(keyboard.Key.shift)
            trKeyboard.release(keyboard.Key.space)


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
