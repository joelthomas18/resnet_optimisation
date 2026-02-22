import tensorflow as tf
import numpy as np
import cv2
import time
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ==============================
# SELECT MODEL TO TEST
# ==============================

MODEL_TYPE = "baseline"  
# Options: "baseline", "pruned", "int8"

BASELINE_PATH = "models/baseline/resnet_baseline.keras"
PRUNED_PATH = "models/pruned/resnet_pruned_50.keras"
INT8_PATH = "models/quantized/resnet50_pruned_50_int8.tflite"

DATASET_PATH = "data/coco/classification"
IMG_SIZE = (224, 224)

# ==============================
# LOAD CLASS NAMES
# ==============================

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=32
)

class_names = dataset.class_names

# ==============================
# LOAD MODEL
# ==============================

if MODEL_TYPE == "baseline":
    model = tf.keras.models.load_model(BASELINE_PATH)

elif MODEL_TYPE == "pruned":
    model = tf.keras.models.load_model(PRUNED_PATH)

elif MODEL_TYPE == "int8":
    interpreter = tf.lite.Interpreter(model_path=INT8_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# ==============================
# START WEBCAM
# ==============================

cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    img = cv2.resize(frame, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.array(img_rgb, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    if MODEL_TYPE in ["baseline", "pruned"]:
        preds = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(preds)]

    elif MODEL_TYPE == "int8":
        scale, zero_point = input_details[0]['quantization']
        int8_input = img_array / scale + zero_point
        int8_input = int8_input.astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], int8_input)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = class_names[np.argmax(output)]

    end_time = time.time()
    latency = (end_time - start_time) * 1000
    fps = 1000 / latency

    # Display text
    cv2.putText(frame, f"Prediction: {predicted_class}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0,255,0), 2)

    cv2.putText(frame, f"Latency: {latency:.2f} ms | FPS: {fps:.2f}", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0,255,255), 2)

    cv2.imshow("Live Surveillance Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()