import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
MAX_IMAGES_PER_CLASS = 500
# =========================
# Configuration
# =========================

DATASET_DIR = "dataset_dev2"
MODEL_DIR = "model"

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.20
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("ASL SIGN LANGUAGE CNN TRAINING")
print("=" * 60)
print("TensorFlow:", tf.__version__)
print("Dataset:", DATASET_DIR)



print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("=" * 60)


# =========================
# Data preparation
# =========================

# =========================
# Data preparation
# =========================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=VALIDATION_SPLIT,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=False,
    brightness_range=(0.8, 1.2)
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=VALIDATION_SPLIT
)

print("\nLoading training dataset...")

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

validation_generator = validation_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)


print("\nClasses detected:")
print(train_generator.class_indices)

print("\nTraining images:", train_generator.samples)
print("Validation images:", validation_generator.samples)


# =========================
# CNN Model
# =========================

print("\nBuilding CNN model...")

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(26, activation="softmax")
])


# =========================
# Compile
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully.")
model.summary()


# =========================
# Callbacks
# =========================

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "asl_cnn_best.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    mode="max",
    verbose=1
)


# =========================
# Training
# =========================

print("\nStarting training...")
print("Watch the epoch progress below.")
print("=" * 60)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        reduce_lr,
        early_stopping
    ]
)


# =========================
# Save model
# =========================

final_model_path = os.path.join(
    MODEL_DIR,
    "asl_cnn_model.keras"
)

model.save(final_model_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Final model:", final_model_path)
print("Best model:", os.path.join(MODEL_DIR, "asl_cnn_best.keras"))
print("=" * 60)