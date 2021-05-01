import cv2
from pynput import mouse, keyboard

trKeyboard = keyboard.Controller()


def print_recognition_text(confidence, gesture, img):
    cv2.putText(img, 'confidence: ' + str(confidence), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, 1)
    cv2.putText(img, 'gesture: ' + gesture, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, 1)


def do_control(index):
    if index == 0:
        trKeyboard.press(keyboard.Key.alt)
        trKeyboard.press(keyboard.Key.tab)
        trKeyboard.release(keyboard.Key.alt)
        trKeyboard.release(keyboard.Key.tab)
    elif index == 1:
        trKeyboard.press(keyboard.Key.ctrl)
        trKeyboard.press('-')
        trKeyboard.release(keyboard.Key.ctrl)
        trKeyboard.release('-')
