import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = "dataset_dev"
MODEL_PATH = "model/asl_cnn_best.keras"

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
SEED = 42

print("=" * 60)
print("ASL MODEL EVALUATION")
print("=" * 60)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded:", MODEL_PATH)

# Validation data only
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.20
)

validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# Evaluate
loss, accuracy = model.evaluate(validation_generator, verbose=1)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(f"Loss:     {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predictions
predictions = model.predict(validation_generator, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

class_names = list(validation_generator.class_indices.keys())

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(cm)

# Save results
os.makedirs("logs", exist_ok=True)

with open("logs/evaluation.txt", "w") as f:
    f.write("ASL MODEL EVALUATION\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Validation samples: {len(y_true)}\n")
    f.write(f"Loss: {loss:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(report)
    f.write("\nCONFUSION MATRIX\n")
    f.write("=" * 60 + "\n")
    f.write(np.array2string(cm))

print("\nEvaluation saved to:")
print("logs/evaluation.txt")