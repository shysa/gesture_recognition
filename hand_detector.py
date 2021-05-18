import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands,
                                        self.detection_con, self.track_con)
        self.mpDraw = mp.solutions.drawing_utils

        self.roi = []

    def find(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        h, w, _ = img.shape

        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                self._find_rectangle(h, w, handLMS)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        else:
            self.roi = []

        return img

    def find_points(self, img, id_to_draw=8):
        lmList = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if id == id_to_draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return lmList

    def _find_rectangle(self, h, w, handLMS):
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h

        for lm in handLMS.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y

        top = y_min
        bottom = y_max
        right = x_min - 15
        if right < 0:
            right = 0
        left = x_max + 15
        if left < 0:
            left = 0
        self.roi = [right, top, left, bottom]

    def get_rectangle(self):
        return self.roi
