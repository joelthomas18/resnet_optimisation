import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==============================
# PATHS
# ==============================

DATASET_PATH = "data/coco/classification"

BASELINE_MODEL_PATH = "models/baseline/resnet_baseline.keras"
PRUNED_30_PATH = "models/pruned/resnet_pruned_30.keras"
PRUNED_50_PATH = "models/pruned/resnet_pruned_50.keras"
PRUNED_70_PATH = "models/pruned/resnet_pruned_70.keras"
INT8_PATH = "models/quantized/resnet50_pruned_50_int8.tflite"

CSV_OUTPUT_PATH = "evaluation/full_model_comparison.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# LOAD VALIDATION DATASET
# ==============================

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = val_dataset.map(
    lambda x, y: (preprocess_input(x), y)
)

val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# ==============================
# FUNCTION: Evaluate Keras Model
# ==============================

def evaluate_keras_model(model_path):
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(val_dataset, verbose=0)
    size = os.path.getsize(model_path) / (1024 * 1024)
    return acc, size

# ==============================
# FUNCTION: Evaluate TFLite Model
# ==============================

def evaluate_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = 0

    for images, labels in val_dataset:
        for i in range(images.shape[0]):
            img = images[i:i+1].numpy()

            # Quantize input
            scale, zero_point = input_details[0]['quantization']
            img = img / scale + zero_point
            img = img.astype(np.int8)

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output)

            if pred == labels[i].numpy():
                correct += 1

            total += 1

    acc = correct / total
    size = os.path.getsize(model_path) / (1024 * 1024)

    return acc, size

# ==============================
# EVALUATION
# ==============================

print("Evaluating Baseline...")
baseline_acc, baseline_size = evaluate_keras_model(BASELINE_MODEL_PATH)

print("Evaluating Pruned 30%...")
pruned30_acc, pruned30_size = evaluate_keras_model(PRUNED_30_PATH)

print("Evaluating Pruned 50%...")
pruned50_acc, pruned50_size = evaluate_keras_model(PRUNED_50_PATH)

print("Evaluating Pruned 70%...")
pruned70_acc, pruned70_size = evaluate_keras_model(PRUNED_70_PATH)

print("Evaluating INT8 Quantized (50%)...")
int8_acc, int8_size = evaluate_tflite_model(INT8_PATH)

# ==============================
# SAVE TO CSV
# ==============================

os.makedirs("evaluation", exist_ok=True)

with open(CSV_OUTPUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Accuracy", "Size (MB)", "Accuracy Drop from Baseline"])

    writer.writerow(["Baseline FP32", round(baseline_acc,4), round(baseline_size,2), 0])

    writer.writerow(["Pruned 30%", 
                     round(pruned30_acc,4), 
                     round(pruned30_size,2), 
                     round(baseline_acc - pruned30_acc,4)])

    writer.writerow(["Pruned 50%", 
                     round(pruned50_acc,4), 
                     round(pruned50_size,2), 
                     round(baseline_acc - pruned50_acc,4)])

    writer.writerow(["Pruned 70%", 
                     round(pruned70_acc,4), 
                     round(pruned70_size,2), 
                     round(baseline_acc - pruned70_acc,4)])

    writer.writerow(["INT8 Quantized (50%)", 
                     round(int8_acc,4), 
                     round(int8_size,2), 
                     round(baseline_acc - int8_acc,4)])

print("\nResults saved to:", CSV_OUTPUT_PATH)

# ==============================
# PRINT SUMMARY
# ==============================

print("\n===== FINAL SUMMARY =====")
print("Baseline:", baseline_acc, "|", baseline_size, "MB")
print("Pruned 30%:", pruned30_acc, "|", pruned30_size, "MB")
print("Pruned 50%:", pruned50_acc, "|", pruned50_size, "MB")
print("Pruned 70%:", pruned70_acc, "|", pruned70_size, "MB")
print("INT8:", int8_acc, "|", int8_size, "MB")