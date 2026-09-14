import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/asl_cnn_best.keras"

IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.60

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("ASL SIGN LANGUAGE REAL-TIME PREDICTION")
print("=" * 60)

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Starting webcam...")
print("Press Q to quit.")
print("=" * 60)

# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror webcam for natural interaction
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    prediction_text = "No hand detected"
    confidence_text = ""

    # ========================================================
    # HAND DETECTION
    # ========================================================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # ----------------------------------------------------
        # Get bounding box from landmarks
        # ----------------------------------------------------

        h, w, _ = frame.shape

        x_coordinates = [
            int(landmark.x * w)
            for landmark in hand_landmarks.landmark
        ]

        y_coordinates = [
            int(landmark.y * h)
            for landmark in hand_landmarks.landmark
        ]

        x_min = max(min(x_coordinates) - 20, 0)
        y_min = max(min(y_coordinates) - 20, 0)

        x_max = min(max(x_coordinates) + 20, w)
        y_max = min(max(y_coordinates) + 20, h)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Crop hand
        # ----------------------------------------------------

        hand_crop = frame[y_min:y_max, x_min:x_max]

        if hand_crop.size > 0:

            # Resize to CNN input size
            resized = cv2.resize(
                hand_crop,
                (IMG_SIZE, IMG_SIZE)
            )

            # Convert BGR → RGB
            resized = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            # Normalize exactly like training
            normalized = resized.astype(
                np.float32
            ) / 255.0

            # Add batch dimension
            input_image = np.expand_dims(
                normalized,
                axis=0
            )

            # ------------------------------------------------
            # CNN PREDICTION
            # ------------------------------------------------

            predictions = model.predict(
                input_image,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            confidence = float(
                predictions[0][predicted_index]
            )

            predicted_letter = CLASS_NAMES[
                predicted_index
            ]

            # ------------------------------------------------
            # Confidence threshold
            # ------------------------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_text = (
                    f"Prediction: {predicted_letter}"
                )

                confidence_text = (
                    f"Confidence: {confidence * 100:.1f}%"
                )

            else:

                prediction_text = "Low confidence"

                confidence_text = (
                    f"Confidence: {confidence * 100:.1f}%"
                )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (500, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        prediction_text,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        confidence_text,
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ASL Sign Language Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
hands.close()
cv2.destroyAllWindows()

print("\nWebcam closed.")
print("Prediction session ended.")