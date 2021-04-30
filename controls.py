import cv2
from pynput import mouse, keyboard

trKeyboard = keyboard.Controller()


def print_recognition_text(conf, index, classes, img):
    cv2.putText(img, 'confidence: ' + str(conf), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, 1)
    cv2.putText(img, 'gesture: ' + classes[index], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, 1)


def do_control(index):
    if index == 0:
        trKeyboard.press(keyboard.Key.ctrl)
        trKeyboard.press('+')
        trKeyboard.release(keyboard.Key.ctrl)
        trKeyboard.release('+')
    elif index == 1:
        trKeyboard.press(keyboard.Key.ctrl)
        trKeyboard.press('-')
        trKeyboard.release(keyboard.Key.ctrl)
        trKeyboard.release('-')
