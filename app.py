import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)
MODEL_PATH = "model/asl_cnn_best.keras"

IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.60

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]

print("Loading ASL CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        "Model output count does not match CLASS_NAMES: "
        f"{model.output_shape[-1]} != {len(CLASS_NAMES)}"
    )

print("Model loaded successfully.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# GLOBAL STATE
# ============================================================

latest_prediction = {
    "letter": None,
    "confidence": 0.0,
    "status": "No hand detected"
}

recognized_text = ""

prediction_history = []

STABLE_FRAMES = 8
RELEASE_FRAMES = 5

last_added_letter = None
hand_release_counter = 0


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame):

    global latest_prediction
    global prediction_history
    global last_added_letter
    global hand_release_counter

    # Flip only for natural webcam interaction
    display_frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        display_frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    prediction = None
    confidence = 0.0
    status = "No hand detected"

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = display_frame.shape

        hand_points = [
            (
                int(landmark.x * w),
                int(landmark.y * h)
            )
            for landmark in hand_landmarks.landmark
        ]

        # ------------------------------------------------------------
        # Create a square crop around the hand
        # ------------------------------------------------------------

        x_min = min(x for x, y in hand_points)
        x_max = max(x for x, y in hand_points)
        y_min = min(y for x, y in hand_points)
        y_max = max(y for x, y in hand_points)

        hand_width = x_max - x_min
        hand_height = y_max - y_min

        # Make the crop square using the larger dimension
        # Scale the margin with the hand so different camera resolutions
        # produce comparable framing and keep fingertips inside the crop.
        hand_extent = max(hand_width, hand_height)
        padding = max(20, int(hand_extent * 0.25))
        crop_size = hand_extent + (2 * padding)

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        square_x_min = center_x - crop_size // 2
        square_y_min = center_y - crop_size // 2
        square_x_max = square_x_min + crop_size
        square_y_max = square_y_min + crop_size

        # Keep the square inside the camera frame
        if square_x_min < 0:
            square_x_max -= square_x_min
            square_x_min = 0

        if square_y_min < 0:
            square_y_max -= square_y_min
            square_y_min = 0

        if square_x_max > w:
            shift = square_x_max - w
            square_x_min -= shift
            square_x_max = w

        if square_y_max > h:
            shift = square_y_max - h
            square_y_min -= shift
            square_y_max = h

        square_x_min = max(square_x_min, 0)
        square_y_min = max(square_y_min, 0)

        hand_crop = display_frame[
            square_y_min:square_y_max,
            square_x_min:square_x_max
        ]

        # Draw the square detection box
        cv2.rectangle(
            display_frame,
            (square_x_min, square_y_min),
            (square_x_max, square_y_max),
            (0, 255, 0),
            2
        )

        mp_drawing.draw_landmarks(
            display_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        if hand_crop.size > 0:

            # Preprocess clean hand image
            resized = cv2.resize(
                hand_crop,
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_AREA
            )

            rgb_crop = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            normalized = (
                rgb_crop.astype("float32") / 255.0
            )

            input_image = np.expand_dims(
                normalized,
                axis=0
            )

            predictions = model.predict(
                input_image,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(predictions)
            )

            confidence = float(
                predictions[predicted_index]
            )

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction = CLASS_NAMES[
                    predicted_index
                ]

                status = "Detected"

                prediction_history.append(
                    prediction
                )

                if len(prediction_history) > STABLE_FRAMES:
                    prediction_history.pop(0)

                if (
                    len(prediction_history)
                    == STABLE_FRAMES
                    and len(set(prediction_history)) == 1
                ):

                    stable_letter = (
                        prediction_history[0]
                    )

                    if (
                        last_added_letter is None
                        or stable_letter != last_added_letter
                    ):

                        add_letter(
                            stable_letter
                        )

                        last_added_letter = (
                            stable_letter
                        )

                        prediction_history.clear()

                hand_release_counter = 0

            else:

                status = "Low confidence"
                prediction_history.clear()

    else:

        prediction_history.clear()

        hand_release_counter += 1

        if hand_release_counter >= RELEASE_FRAMES:

            last_added_letter = None

    latest_prediction = {
        "letter": prediction,
        "confidence": round(
            confidence * 100,
            1
        ),
        "status": status
    }

    # Camera overlay
    cv2.rectangle(
        display_frame,
        (10, 10),
        (430, 100),
        (0, 0, 0),
        -1
    )

    display_letter = (
        prediction
        if prediction
        else "-"
    )

    cv2.putText(
        display_frame,
        f"Prediction: {display_letter}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Confidence: {confidence * 100:.1f}%",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return display_frame


# ============================================================
# ADD LETTER
# ============================================================

def add_letter(letter):

    global recognized_text

    if len(recognized_text) >= 200:
        return

    recognized_text += letter


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open webcam.")

        return

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = process_frame(frame)

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

    camera.release()


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/prediction")
def prediction():

    return jsonify({
        **latest_prediction,
        "text": recognized_text
    })


@app.route("/clear_text", methods=["POST"])
def clear_text():

    global recognized_text
    global prediction_history
    global last_added_letter

    recognized_text = ""

    prediction_history.clear()

    last_added_letter = None

    return jsonify({
        "success": True,
        "text": ""
    })


@app.route("/space", methods=["POST"])
def add_space():

    global recognized_text

    if (
        recognized_text
        and not recognized_text.endswith(" ")
    ):
        recognized_text += " "

    return jsonify({
        "success": True,
        "text": recognized_text
    })


@app.route("/backspace", methods=["POST"])
def backspace():

    global recognized_text

    recognized_text = recognized_text[:-1]

    return jsonify({
        "success": True,
        "text": recognized_text
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ASL SIGN LANGUAGE WEB APPLICATION")
    print("=" * 60)
    print("Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )