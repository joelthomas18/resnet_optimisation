import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import csv

# ==============================
# PATH CONFIGURATION
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "data", "coco", "classification")
BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "models", "baseline", "resnet_baseline.keras")
PRUNED_MODEL_DIR = os.path.join(BASE_DIR, "models", "pruned")
RESULTS_FILE = os.path.join(BASE_DIR, "evaluation", "pruning_results.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6
SPARSITY_LEVELS = [0.30, 0.50, 0.70]

# ==============================
# LOAD DATASET
# ==============================

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# ==============================
# BASELINE EVALUATION
# ==============================

baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
baseline_loss, baseline_acc = baseline_model.evaluate(val_dataset)

baseline_size = os.path.getsize(BASELINE_MODEL_PATH) / (1024 * 1024)

print(f"\nBaseline Accuracy: {baseline_acc:.4f}")
print(f"Baseline Size: {baseline_size:.2f} MB")

# ==============================
# PRUNING EXPERIMENTS
# ==============================

results = []

for sparsity in SPARSITY_LEVELS:

    print("\n==============================")
    print(f"Running Pruning at {int(sparsity*100)}% Sparsity")
    print("==============================")

    model = tf.keras.models.load_model(BASELINE_MODEL_PATH)

    num_images = len(train_dataset)
    end_step = num_images * EPOCHS

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=sparsity,
            begin_step=0,
            end_step=end_step
        )
    }

    # Apply pruning to entire model
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        **pruning_params
    )

    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep()
    ]

    pruned_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Strip pruning wrappers
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs(PRUNED_MODEL_DIR, exist_ok=True)

    pruned_loss, pruned_acc = final_model.evaluate(val_dataset)

    model_name = f"resnet_pruned_{int(sparsity*100)}.keras"
    model_path = os.path.join(PRUNED_MODEL_DIR, model_name)

    final_model.save(model_path)

    pruned_size = os.path.getsize(model_path) / (1024 * 1024)

    acc_drop = baseline_acc - pruned_acc

    print(f"Pruned Accuracy: {pruned_acc:.4f}")
    print(f"Accuracy Drop: {acc_drop:.4f}")

    results.append([
        int(sparsity*100),
        round(pruned_acc, 4),
        round(pruned_size, 2),
        round(acc_drop, 4)
    ])

# ==============================
# SAVE RESULTS
# ==============================

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Sparsity (%)", "Accuracy", "Model Size (MB)", "Accuracy Drop"])
    writer.writerows(results)

print("\nExperiment Completed!")
print("Results saved in evaluation/pruning_results.csv")
