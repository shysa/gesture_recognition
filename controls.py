import cv2
from pynput import mouse, keyboard

trKeyboard = keyboard.Controller()
trMouse = mouse.Controller()

xp, yp = 0, 0

prev_command = -1


def print_recognition_text(confidence, gesture, img):
    cv2.putText(img, 'Gesture: ' + gesture, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (216, 180, 0), 1, 1)
    cv2.putText(img, 'Confidence: ' + str(confidence[0]), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.9, (216, 180, 0), 1, 1)


def print_text(text, img):
    cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (216, 180, 0), 1, 1)


def do_control(index):
    global prev_command

    # do only the first recognized gesture
    if prev_command != index or prev_command == 10:
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

        # Gesture and action: Mouse click
        elif index == 10:
            trMouse.press(mouse.Button.left)
            trMouse.release(mouse.Button.left)

    prev_command = index


def move_mouse(lmList, k):
    global xp, yp

    if len(lmList) != 0:
        cx, cy = lmList[8][1], lmList[8][2]

        if xp == 0 and yp == 0:
            xp, xy = cx, cy

        if abs(xp - cx) > 4:
            xp = cx
        if abs(yp - cy) > 4:
            yp = cy

        x, y = xp * k[0], yp * k[1]
        trMouse.position = (x, y)
