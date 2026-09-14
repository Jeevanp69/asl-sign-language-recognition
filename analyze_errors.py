import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

DATASET_DIR = "dataset_dev"
MODEL_PATH = "model/asl_cnn_best.keras"

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
SEED = 42

os.makedirs("logs", exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.20
)

generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

class_names = list(generator.class_indices.keys())

predictions = model.predict(generator, verbose=1)

y_true = generator.classes
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_true, y_pred)

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

plt.figure(figsize=(12, 10))

plt.imshow(cm, interpolation="nearest")
plt.title("ASL Validation Confusion Matrix")
plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        if cm[i, j] > 0:
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

plt.tight_layout()

plt.savefig(
    "logs/confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# Most confused class pairs
# --------------------------------------------------

pairs = []

for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            pairs.append(
                (
                    cm[i, j],
                    class_names[i],
                    class_names[j]
                )
            )

pairs.sort(reverse=True)

print("\n" + "=" * 60)
print("MOST COMMON MISCLASSIFICATIONS")
print("=" * 60)

for count, actual, predicted in pairs[:20]:
    print(
        f"Actual {actual} -> Predicted {predicted}: {count}"
    )

print("\nConfusion matrix saved to:")
print("logs/confusion_matrix.png")