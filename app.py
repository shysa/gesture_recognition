import streamlit as st
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from gesture_detector import GestureDetector

MODEL = './model'
CLASSES = ["Нет жеста", "Слайд вниз", "Слайд вверх", "Свайп влево", "Свайп вправо"]


def main():
    st.set_page_config(layout="wide")
    st.header("Система распознавания жестов с использованием библиотеки OpenCV")

    col1, col2, col3 = st.beta_columns([2, .5, 3])

    table = col3.table()
    plot = col3.empty()

    #
    # RESULTS PART
    # TABLE OF PREDICTIONS AND PLOT
    #
    with col3:
        df = pd.DataFrame(
            [[0.0, 0.0, 0.0, 0.0, 0.0]],
            columns=CLASSES)

        df.index = [""] * len(df)
        table.table(df.style)

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.bar([0, 1, 2, 3, 4], [1, 0, 0, 0, 0])
        plot.pyplot(fig)

    #
    # CAMERA PART WITH PREVIEW AND RECOGNIZER
    #
    camera = cv2.VideoCapture(0)

    recognizer = GestureDetector()
    recognizer.get_classes()

    with col1:
        FRAME_WINDOW = st.image([])
        FRAME_WINDOW.image('./app/stop_camera.png')

        start = st.button("Старт")
        stop = st.button("Стоп")

        if start:
            while True:
                # get the frame and show it
                _, frame = camera.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(cv2.flip(frame, 1))

                # get results of model prediction
                result = recognizer.find_gesture(frame, full_pred=True)

                if result is not None:
                    (confidence, _, _) = result

                    # update confidence table
                    ndf = pd.DataFrame(
                        [confidence],
                        columns=CLASSES)
                    ndf.index = [""] * len(ndf)
                    table.table(ndf.style.highlight_max(axis=1))

                    # update plot hist
                    nfig, nax = plt.subplots(figsize=(5, 2))
                    nax.bar([0, 1, 2, 3, 4], confidence)
                    plot.pyplot(nfig)

                if stop:
                    break

        camera.release()


main()
