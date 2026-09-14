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
CNN Architecture

The classifier uses a lightweight convolutional neural network designed for static ASL alphabet classification.

Input: 64 x 64 x 3
        |
Conv2D - 32 filters
        |
MaxPooling
        |
Conv2D - 64 filters
        |
MaxPooling
        |
Conv2D - 128 filters
        |
MaxPooling
        |
Flatten
        |
Dense - 256
        |
Dropout - 0.5
        |
Dense - 26
        |
Softmax

Total trainable parameters: 1,279,834

Model Evaluation

The current primary model was evaluated on a development validation set containing 2,600 images, representing 100 validation images for each of the 26 alphabet classes.

Metric	Result
Accuracy	83.46%
Macro Precision	85.16%
Macro Recall	83.46%
Macro F1	83.11%

These results represent the rebuilt development model and are not the historical results reported in the original academic project.

Inference Benchmark

CNN inference was benchmarked locally on a CPU system.

Metric	Result
Average inference time	73.39 ms
Median inference time	70.81 ms
Minimum inference time	62.85 ms
Maximum inference time	166.84 ms
Estimated CNN throughput	13.63 FPS

Performance may vary depending on hardware and system load.

Dataset

The project uses an ASL alphabet image dataset organized into 26 class folders:

A/
B/
C/
...
Z/

For development, a subset containing 500 images per class was prepared.

This produced:

13,000 development images
10,400 training images
2,600 validation images

The complete dataset is intentionally excluded from this repository because of its size and dataset licensing/distribution considerations.

Error Analysis

The project includes an error-analysis script that generates a confusion matrix and identifies frequent misclassifications.

Some frequent errors in the current model include:

S -> T
Y -> T
V -> W
R -> U
L -> O
X -> T
R -> V
H -> G
W -> R
Y -> X

These errors indicate that visually similar hand configurations remain challenging for the current CNN.

Run:

python analyze_errors.py
Project Structure
asl-sign-language-recognition/
│
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── analyze_errors.py
├── benchmark.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── app.js

The trained model, datasets, logs, and virtual environment are intentionally excluded from version control.

Tech Stack
Python 3.10
TensorFlow 2.15.1
OpenCV
MediaPipe
Flask
NumPy
scikit-learn
pyttsx3
HTML
CSS
JavaScript
Installation
1. Clone the repository
git clone https://github.com/Jeevanp69/asl-sign-language-recognition.git
cd asl-sign-language-recognition
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment

Windows PowerShell:

venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
Running the Application

The trained model must be available at:

model/asl_cnn_best.keras

Then run:

python app.py

Open:

http://127.0.0.1:5000

Allow webcam access and place your hand inside the camera view.

Training

The CNN can be trained using:

python train.py

Training configuration and dataset paths are defined inside train.py.

The dataset is not included in this repository.

Evaluation

Run:

python evaluate.py

The evaluation script reports:

Validation accuracy
Classification metrics
Per-class performance

Evaluation output is stored locally in the logs/ directory.

Performance Benchmark

Run:

python benchmark.py

This measures CNN inference performance on the local system.

Limitations
Only static ASL alphabet gestures are currently recognized.
Dynamic signs are not supported.
Non-manual signals are not recognized.
Full ASL grammar and sentence-level interpretation are outside the current scope.
Recognition performance can vary with lighting, camera quality, hand position, background, and viewing conditions.
Visually similar gestures can occasionally be confused.
The current model is a development-stage classifier rather than a production-ready sign-language translation system.
Future Improvements
Larger and more diverse training data
Improved robustness across lighting and backgrounds
Better handling of difficult letter pairs
Transfer learning using a lightweight pretrained vision model
GPU acceleration
Dynamic gesture recognition
Sentence-level sign-language understanding
Cloud or mobile deployment
Author

Jeevan Puppala

Built as an AI/ML computer vision portfolio project.


### Then update GitHub

Since you already committed the old README, after replacing it locally run:

```powershell
git add README.md
git commit -m "Improve project documentation"
git push
