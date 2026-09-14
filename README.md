# ASL Sign Language Recognition to Text and Speech

A real-time computer vision application that recognizes American Sign Language (ASL) alphabet gestures using a convolutional neural network (CNN) and MediaPipe, converts recognized gestures into text, and provides text-to-speech output through a Flask web interface.

## Features

- Real-time webcam-based ASL alphabet recognition
- Recognition of 26 static ASL alphabet letters (A-Z)
- MediaPipe hand detection and landmark tracking
- CNN-based image classification
- Confidence-based prediction filtering
- Stable-frame prediction to reduce accidental letters
- Automatic recognized-text formation
- Space, backspace, and clear controls
- Text-to-speech using `pyttsx3`
- Flask web interface
- Real-time prediction and confidence display

## Demo

The application provides a browser-based interface with:

- Live webcam feed
- Hand detection and bounding box
- Predicted ASL letter
- Prediction confidence
- Recognized text
- Text-to-speech functionality

## System Architecture

```text
Webcam
   |
   v
OpenCV Frame Capture
   |
   v
MediaPipe Hand Detection
   |
   v
Square Hand Crop
   |
   v
64 x 64 RGB Preprocessing
   |
   v
CNN Classifier
   |
   v
A-Z Prediction + Confidence
   |
   v
Stable Prediction Filtering
   |
   +------------------+
   |                  |
   v                  v
Recognized Text     Text-to-Speech
