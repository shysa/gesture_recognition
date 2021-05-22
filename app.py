import streamlit as st
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import controls
from gesture_detector import GestureDetector
from hand_detector import HandDetector


MODEL = './model'
CLASSES = ["Нет жеста", "Слайд вниз", "Слайд вверх", "Свайп влево", "Свайп вправо"]

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def draw_recognition_results(table, plot, fig, ax, conf=None, view="Таблица"):
    if conf is None:
        conf = [0.0, 0.0, 0.0, 0.0, 0.0]

    df = pd.DataFrame(
        [conf],
        columns=CLASSES)
    df.index = [""] * len(df)

    if view == "Таблица" or view == "Таблица и график":
        if any(item == 0.0 for item in conf):
            table.table(df)
        else:
            table.table(df.style.highlight_max(axis=1))

    if view == "График" or view == "Таблица и график":
        plt.cla()
        ax.bar(CLASSES, conf)
        plot.pyplot(fig)


def main():
    # ------------------------------------------
    # DEFINITIONS
    # ------------------------------------------
    st.set_page_config(layout="wide")
    st.header("Система распознавания жестов с использованием библиотеки OpenCV")

    col1, col2, col3 = st.beta_columns([3, .3, 4])
    table = col3.empty()
    plot = col3.empty()
    FRAME_WINDOW = col1.image([])

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.tick_params(axis='x', labelsize=6)
    ax.bar(CLASSES, [1, 0, 0, 0, 0])

    btn1, btn2 = st.sidebar.beta_columns([3, 1])
    start = btn1.button("Старт")
    stop = btn2.button("Стоп")

    st.sidebar.write("#")

    result_view_select = st.sidebar.selectbox(
        "Отображать результаты распознавания",
        ["Таблица", "График", "Таблица и график"],
        0
    )

    st.sidebar.write("#")

    control_allow = st.sidebar.radio(
        "Разрешить режим управления?",
        ["Нет, только распознавание", "Разрешить с ограничениями", "Полный доступ"]
    )

    # ------------------------------------------
    # CAMERA RECOGNIZER AND HAND DETECTOR
    # ------------------------------------------
    camera = cv2.VideoCapture(0)

    recognizer = GestureDetector()
    recognizer.get_classes()
    gesture_index = 0

    detector = HandDetector(max_hands=1, track_con=0.85)
    mouse_mode = False
    k = [1920 / 640, 1080 / 480]

    # for pause after mouse mode
    skipped_frames = 0

    # ------------------------------------------
    # APP
    # ------------------------------------------
    with col3:
        draw_recognition_results(table, plot, fig, ax, view=result_view_select)

    with col1:
        FRAME_WINDOW.image('./app/stop_camera.png')

        msg = st.text("")
        action = st.text("")

        if start:
            while True:
                # get the frame and show it
                _, frame = camera.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(cv2.flip(frame, 1))

                if control_allow == "Нет, только распознавание":
                    result = recognizer.find_gesture(frame, full_pred=True)

                    if result is not None:
                        (confidence, _, _) = result

                        # update confidence results
                        draw_recognition_results(table, plot, fig, ax, confidence, result_view_select)
                else:
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

                            if control_allow == "Полный доступ":
                                controls.do_control(10)
                            else:
                                action.text("Клик")

                            cv2.circle(img, (click_target_x, click_target_y), 10, (91, 94, 255), cv2.FILLED)
                            cv2.circle(img, (thumb_x, thumb_y), 10, (91, 94, 255), cv2.FILLED)

                        # draw cursor point on index finger and for click
                        index_x, index_y = lmlist[8][1], lmlist[8][2]
                        cv2.circle(img, (index_x, index_y), 5, (91, 94, 255), cv2.FILLED)

                        cv2.circle(img, (click_target_x, click_target_y), 10, (91, 94, 255), thickness=1)
                        cv2.circle(img, (thumb_x, thumb_y), 10, (91, 94, 255), thickness=1)
                        cv2.line(img, (click_target_x, click_target_y), (thumb_x, thumb_y), (91, 94, 255), thickness=1)

                        controls.move_mouse(lmlist, k)
                        msg.text("Включен режим мыши")

                    else:
                        if skipped_frames >= 20:
                            # get results of model prediction
                            result = recognizer.find_gesture(frame, full_pred=True)

                            if result is not None:
                                (confidence, _, gesture_index) = result

                                if control_allow == "Полный доступ":
                                    controls.do_control(gesture_index)
                                else:
                                    action.text("Действие для " + CLASSES[gesture_index])

                                # update confidence results
                                draw_recognition_results(table, plot, fig, ax, confidence, result_view_select)
                            else:
                                msg.text("Включается режим распознавания, подождите")

                    if stop:
                        break

        camera.release()


main()
