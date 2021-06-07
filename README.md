## Gesture recognition and hand tracking for gesture control
###### Diploma
Deep learning project with keras/tensorFlow. Based on The 20BN-jester Dataset V1 and MediaPipe Hands.
Dataset can be downloaded [here](https://20bn.com/datasets/jester). Model uses 3 3D-convolutional layers and 2 LSTM layers.


### Versions
- python 3.7
- keras 2.4.3
- tensorflow 2.4.1
- opencv-python 4.5.1
- mediapipe 0.8.3.1

*Higly recommend using GPU-computation:*
- tensorflow-gpu 2.4.1
- CUDA Toolkit 11.3 + cuDNN 8.2.0 (more information [here](https://developer.nvidia.com/cuda-toolkit "CUDA Toolkit") and [here](https://developer.nvidia.com/cudnn "cuDNN"))

### Structure
- `data` to store dataset files. Advisable to group dataset folders into gesture named folders
- `train.ipynb` for training with Jupyter Notebook and save model into dir `model`
- `controls.py` contained funcs for a keyboard and mouse control
- `gesture_detector.py` contained funcs for detecting mouse gestures and class for trained model for recognition
- `hand_detector.py` to find landmarks on frame with MediaPipe Hands
