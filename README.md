# Computer Vision Facial & Behavioral Analysis Module

A modular real-time Computer Vision module built with **Python**, **OpenCV**, **MediaPipe**, and **PyTorch (ViT)** for facial detection, expression classification, eye/gaze tracking, and 3D head pose estimation.

## Core Modules (`computer_vision/`)

- **`face_detector.py`**: Real-time multi-face detection and 478-point 3D facial landmark extraction using MediaPipe `FaceLandmarker`.
- **`expression_recognizer.py`**: Visual facial expression classification across 7 classes (*Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral*) using a pretrained Vision Transformer (`trpakov/vit-face-expression`) with MediaPipe blendshape fallback.
- **`eye_tracker.py`**: Eye Aspect Ratio (EAR) calculation, rolling blink frequency detection (blinks/min), and iris-based gaze direction estimation (*left, right, center, up, down*).
- **`head_pose.py`**: 3D head orientation estimation (*Yaw, Pitch, Roll*) via OpenCV `solvePnP` to detect off-camera head turns.

## Tech Stack

- **Computer Vision:** OpenCV, MediaPipe
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Scientific Computing:** NumPy
