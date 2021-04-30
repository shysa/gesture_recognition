import cv2
import mediapipe as mp
from pynput import mouse, keyboard
from hand_detector import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1, track_con=0.85)
trMouse = mouse.Controller()

xp, yp = 0, 0

while True:
    _, img = cap.read()
    img = detector.find(img, draw=False)
    lmList = detector.find_points(img)

    if len(lmList) != 0:
        cx, cy = lmList[8][1], lmList[8][2]

        if xp == 0 and yp == 0:
            xp, xy = cx, cy

        x, y = (xp - cx) * 4, (cy - yp) * 3.5
        trMouse.move(x, y)
        xp, yp = cx, cy

    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


