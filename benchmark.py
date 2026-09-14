import time
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/asl_cnn_best.keras"

IMG_SIZE = 64
NUM_WARMUP = 20
NUM_RUNS = 100

print("=" * 60)
print("ASL CNN PERFORMANCE BENCHMARK")
print("=" * 60)

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")

# Random input with the same shape as the real pipeline
sample = np.random.random(
    (1, IMG_SIZE, IMG_SIZE, 3)
).astype(np.float32)

# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------

print("\nWarming up model...")

for _ in range(NUM_WARMUP):

    model.predict(
        sample,
        verbose=0
    )

# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

print("Running benchmark...")

times = []

for _ in range(NUM_RUNS):

    start = time.perf_counter()

    model.predict(
        sample,
        verbose=0
    )

    end = time.perf_counter()

    times.append(
        (end - start) * 1000
    )

times = np.array(times)

average_ms = np.mean(times)
median_ms = np.median(times)
min_ms = np.min(times)
max_ms = np.max(times)

fps = 1000 / average_ms

# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

parameters = model.count_params()

print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print(
    f"Model parameters : {parameters:,}"
)

print(
    f"Average inference: {average_ms:.2f} ms"
)

print(
    f"Median inference : {median_ms:.2f} ms"
)

print(
    f"Minimum inference: {min_ms:.2f} ms"
)

print(
    f"Maximum inference: {max_ms:.2f} ms"
)

print(
    f"Estimated CNN FPS: {fps:.2f}"
)

print("=" * 60)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

with open(
    "logs/benchmark.txt",
    "w"
) as file:

    file.write(
        "ASL CNN PERFORMANCE BENCHMARK\n"
    )

    file.write("=" * 60 + "\n")

    file.write(
        f"Model parameters: {parameters:,}\n"
    )

    file.write(
        f"Average inference: {average_ms:.2f} ms\n"
    )

    file.write(
        f"Median inference: {median_ms:.2f} ms\n"
    )

    file.write(
        f"Minimum inference: {min_ms:.2f} ms\n"
    )

    file.write(
        f"Maximum inference: {max_ms:.2f} ms\n"
    )

    file.write(
        f"Estimated CNN FPS: {fps:.2f}\n"
    )

print("\nResults saved to:")
print("logs/benchmark.txt")