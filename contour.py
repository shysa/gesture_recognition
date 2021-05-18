import cv2
from hand_detector import HandDetector
from segment import run_avg, segment

cap = cv2.VideoCapture(0)
detector = HandDetector(max_hands=1, track_con=0.85)

trackMode = False

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

    # create cache img to erase roi rectangles
    copy = img.copy()

    # convert the img to grayscale and blur it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # CALIBRATION STEP
    # to get the background, keep looking till a threshold is reached
    # so that our weighted average model gets calibrated
    if num_frames < 30:
        run_avg(gray, accumWeight)
        if num_frames == 1:
            print("[STATUS] please wait! calibrating...")
        elif num_frames == 29:
            print("[STATUS] calibration successful...")
    else:
        # detect hand joints
        img = detector.find(img)

        # find coord for joints for CLICK gesture tracking
        lmlist = detector.find_points(img)

        # get ROI coords rectangle
        coords_roi = detector.get_rectangle()

        if len(lmlist) != 0 and len(coords_roi) != 0:
            # get the ROI and draw it on frame
            roi = gray[coords_roi[1]:coords_roi[3], coords_roi[0]:coords_roi[2]]
            cv2.rectangle(img, (coords_roi[0], coords_roi[1]), (coords_roi[2], coords_roi[3]), (0, 255, 0), 2)

            # segment the hand region
            hand = segment(roi, coords_roi)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and segmented region
                (thresholded, segmented) = hand
                cvxHull = cv2.convexHull(segmented)

                # draw the segmented region and display the frame
                cv2.drawContours(img, [segmented], -1, (0, 0, 255))
                # cv2.drawContours(img, [cvxHull], -1, (0, 255, 0))

                # show the thresholded image
                cv2.imshow("Thesholded", cv2.flip(thresholded, 1))

            # get some coords for CLICK checking
            thumb_x, thumb_y = lmlist[4][1], lmlist[4][2]
            click_target_x, click_target_y = lmlist[5][1], lmlist[5][2]

            click_x = abs(thumb_x - click_target_x)
            click_y = abs(thumb_y - click_target_y)

            if click_x <= 30 and click_y <= 30:
                print("click")
        else:
            img = copy.copy()

    # increment the number of frames
    num_frames += 1

    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
