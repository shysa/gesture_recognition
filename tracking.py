import cv2
import mediapipe as mp
from pynput import mouse, keyboard
from hand_detector import HandDetector
from segment import run_avg, segment

cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1, track_con=0.85)
trMouse = mouse.Controller()

xp, yp = 0, 0

# initialize accumulated weight
accumWeight = 0.5

# region of interest (ROI) coordinates
top, right, bottom, left = 10, 50, 225, 290
roiArea = (bottom - top) * (left - right)

# calibration indicator
calibrated = False

# initialize num of frames for calibration
num_frames = 0

while True:
    # get the current frame
    _, img = cap.read()

    # get the ROI for gesture
    roi = img[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # CALIBRATION STEP
    # to get the background, keep looking till a threshold is reached
    # so that our weighted average model gets calibrated
    if num_frames < 30:
        run_avg(gray, accumWeight)
        if num_frames == 1:
            print("[STATUS] please wait! calibrating...")
        elif num_frames == 29:
            print("[STATUS] calibration successfull...")
    else:
        # AFTER CALIBRATION WE SHOULD CHECK
        # IF HAND IN ROI THEN DO GESTURE RECOGNITION
        # ELSE WE TRACK HAND FOR CURSOR
        #
        # segment the hand region
        hand = segment(gray)

        # check whether hand region is segmented
        if hand is not None:
            # if yes, unpack the thresholded image and
            # segmented region
            (thresholded, segmented) = hand

            if cv2.contourArea(segmented)/roiArea >= 0.05:
                # draw the segmented region and display the frame
                cv2.drawContours(img, [segmented + (right, top)], -1, (0, 0, 255))

                # show the thresholded image
                cv2.imshow("Thesholded", cv2.flip(thresholded, 1))
        else:
            img = detector.find(img, draw=False)
            lmList = detector.find_points(img)

            if len(lmList) != 0:
                cx, cy = lmList[8][1], lmList[8][2]

                if xp == 0 and yp == 0:
                    xp, xy = cx, cy

                x, y = (xp - cx) * 4, (cy - yp) * 3.5
                trMouse.move(x, y)
                xp, yp = cx, cy

    # draw the segmented hand
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    # increment the number of frames
    num_frames += 1

    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


