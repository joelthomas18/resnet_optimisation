import os
import tensorflow as tf
import numpy as np

# ==============================
# PATH CONFIGURATION
# ==============================

PRUNED_MODEL_PATH = "models/pruned/resnet_pruned_50.keras"
TFLITE_MODEL_PATH = "models/quantized/resnet50_pruned_50_int8.tflite"
DATASET_PATH = "data/coco/classification"

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

# Apply same preprocessing used during training
from tensorflow.keras.applications.resnet50 import preprocess_input

val_dataset = val_dataset.map(
    lambda x, y: (preprocess_input(x), y)
)

val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# ==============================
# LOAD PRUNED MODEL
# ==============================

model = tf.keras.models.load_model(PRUNED_MODEL_PATH)

# Evaluate pruned FP32 model
loss, acc = model.evaluate(val_dataset)
print("\nPruned FP32 Accuracy:", round(acc, 4))

# ==============================
# QUANTIZATION (INT8)
# ==============================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset (VERY IMPORTANT)
def representative_data_gen():
    for images, _ in val_dataset.take(100):
        yield [images]

converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

os.makedirs("models/quantized", exist_ok=True)

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print("\nINT8 Quantized model saved successfully.")

size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
print("Quantized Model Size:", round(size, 2), "MB")

# ==============================
# EVALUATE TFLITE MODEL
# ==============================

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

correct = 0
total = 0

for images, labels in val_dataset:
    for i in range(images.shape[0]):
        img = images[i:i+1]

        # Quantize input properly
        scale, zero_point = input_details[0]['quantization']
        img = img / scale + zero_point
        img = img.numpy().astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output)

        if pred == labels[i].numpy():
            correct += 1
        total += 1

quant_acc = correct / total
print("\nINT8 Quantized Accuracy:", round(quant_acc, 4))