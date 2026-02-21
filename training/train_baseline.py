import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

DATA_DIR = "data/coco/classification"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10  # Phase 1
FINE_TUNE_EPOCHS = 20 # Phase 2

# ==============================
# LOAD DATASET
# ==============================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

num_classes = len(train_ds.class_names)
AUTOTUNE = tf.data.AUTOTUNE

# Enhanced Augmentation for Surveillance
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomContrast(0.2), # Good for different lighting
    layers.RandomZoom(0.1)
])

def prepare_data(ds, augment=False):
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train_dataset = prepare_data(train_ds, augment=True)
val_dataset = prepare_data(val_ds)

# ==============================
# BUILD MODEL
# ==============================
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# PHASE 1: Freeze Base
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# ==============================
# TRAINING
# ==============================
# Stage 1: Train Top Layers Only
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Phase 1: Training the Classification Head...")
model.fit(train_dataset, validation_data=val_dataset, epochs=INITIAL_EPOCHS)

# Stage 2: Fine-Tuning
print("Phase 2: Fine-tuning ResNet Layers...")
base_model.trainable = True
# Freeze early layers, train the rest
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Lower LR for fine-tuning
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]

model.fit(train_dataset, validation_data=val_dataset, 
          epochs=FINE_TUNE_EPOCHS, callbacks=callbacks)

# ==============================
# SAVE
# ==============================
os.makedirs("models/baseline", exist_ok=True)
model.save("models/baseline/resnet_baseline.keras")