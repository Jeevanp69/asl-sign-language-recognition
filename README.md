# ASL Sign Language Recognition to Text and Speech

A real-time computer vision application that recognizes American Sign Language (ASL) alphabet gestures using a CNN and MediaPipe, converts recognized signs into text, and provides text-to-speech output.

## Features

- Real-time webcam-based ASL alphabet recognition
- Recognition of 26 static ASL alphabet letters (A-Z)
- MediaPipe hand detection and landmark tracking
- CNN-based image classification
- Confidence-based prediction filtering
- Stable-frame prediction to reduce accidental letters
- Automatic text formation
- Space, backspace, and clear controls
- Text-to-speech using `pyttsx3`
- Flask web interface
- Real-time prediction and confidence display

## Tech Stack

- Python 3.10
- TensorFlow 2.15.1
- OpenCV
- MediaPipe
- Flask
- NumPy
- scikit-learn
- pyttsx3
- HTML / CSS / JavaScript

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

The classifier uses a lightweight convolutional neural network:

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

Total trainable parameters:

1,279,834

Model Evaluation

The current primary model was evaluated on the development validation set containing 2,600 images.

Metric	Result
Accuracy	83.46%
Macro Precision	85.16%
Macro Recall	83.46%
Macro F1	83.11%
Inference Benchmark

Measured on a CPU system:

Metric	Result
Average inference time	73.39 ms
Median inference time	70.81 ms
Minimum inference time	62.85 ms
Maximum inference time	166.84 ms
Estimated CNN throughput	13.63 FPS

These are measured project results and may vary depending on hardware.

Dataset

The project uses an ASL alphabet image dataset organized into 26 class folders:

A/
B/
C/
...
Z/

For development and evaluation, a subset containing 100 images per class was used for validation.

The complete dataset is intentionally excluded from this repository because of its size and dataset licensing/distribution considerations.

Project Structure
asl-sign-language/
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
├── model/
│   └── .gitkeep
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
└── logs/
Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd asl-sign-language
2. Create a virtual environment
python -m venv venv
3. Activate the environment

Windows PowerShell:

venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
Running the Application

Make sure the trained model is available at:

model/asl_cnn_best.keras

Then run:

python app.py

Open the application in your browser:

http://127.0.0.1:5000

Allow webcam access and place your hand inside the camera view.

Training

The training script can be used to train the CNN on the prepared dataset.

python train.py

Training configuration and dataset paths are defined inside train.py.

Evaluation

Run:

python evaluate.py

This generates:

validation loss
validation accuracy
classification report
confusion matrix information

Results are saved to:

logs/evaluation.txt
Error Analysis

Run:

python analyze_errors.py

The script identifies common misclassifications and generates a confusion matrix.

Some of the most frequent errors in the current model include visually similar letter groups such as:

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
Performance Benchmark

Run:

python benchmark.py

This measures CNN inference performance on the local system.

Limitations
The current model recognizes static ASL alphabet gestures only.
Dynamic signs are not supported.
Non-manual signals are not recognized.
Full ASL grammar and sentence-level interpretation are outside the current scope.
Recognition performance can vary with lighting, camera quality, hand position, background, and viewing conditions.
Similar-looking gestures can occasionally be confused.
Future Improvements
Larger and more diverse training data
Improved robustness across lighting and backgrounds
Better handling of difficult letter pairs
Transfer learning with a lightweight pretrained vision model
GPU acceleration
Dynamic gesture recognition
Sentence-level sign language understanding
Deployment as a cloud or mobile application
Author

Jeevan Puppala

Built as an AI/ML computer vision portfolio project.


### One important correction

Your README should **not** claim:

> 98.5% accuracy

from the old academic report.

Our rebuilt project has a **measured 83.46% validation accuracy**, and that's the number to defend in an interview.

