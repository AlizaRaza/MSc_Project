"""
Stage 1: VGG16 Baseline - Emotion Classification
Dataset: Autism Emotion Recognition Dataset (FER-Autism)
Task: 6-class facial emotion classification (Natural, Fear, Surprise, Sadness, Anger, Joy)
Author: Aliza Raza | Student ID: 250915589
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET_PATH  = "/Users/alizaraza/Downloads/Autism emotion recogition dataset"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-4
RESULTS_DIR   = os.path.expanduser("~/Desktop/MSc_Project/results_stage1")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("STAGE 1: VGG16 BASELINE — Emotion Classification")
print("Dataset: FER-Autism")
print(f"Results will be saved to: {RESULTS_DIR}")
print("=" * 60)

# ── DATA GENERATORS ───────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = len(train_generator.class_indices)
CLASS_NAMES = list(train_generator.class_indices.keys())

print(f"\nClasses ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"Training samples:   {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples:       {test_generator.samples}")

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
print("\nBuilding VGG16 model...")

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all base layers initially
for layer in base_model.layers:
    layer.trainable = False

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"Total parameters: {model.count_params():,}")

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=10,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        os.path.join(RESULTS_DIR, "vgg16_best.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-7, verbose=1)
]

# ── PHASE 1: Train head only ──────────────────────────────────────────────────
print("\n--- Phase 1: Training classification head (base frozen) ---")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# ── PHASE 2: Fine-tune last conv block ───────────────────────────────────────
print("\n--- Phase 2: Fine-tuning last convolutional block (block5) ---")
for layer in base_model.layers:
    if "block5" in layer.name:
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ── COMBINE HISTORIES ─────────────────────────────────────────────────────────
history = {}
for key in history1.history:
    history[key] = history1.history[key] + history2.history[key]

# ── EVALUATE ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
np.save(os.path.join(RESULTS_DIR, "history.npy"), history)
np.save(os.path.join(RESULTS_DIR, "y_true.npy"), y_true)
np.save(os.path.join(RESULTS_DIR, "y_pred.npy"), y_pred)

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle("Stage 1 — VGG16 Emotion Classification (FER-Autism)", fontsize=13)

# Accuracy
axes[0].plot(history["accuracy"], label="Train")
axes[0].plot(history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history["loss"], label="Train")
axes[1].plot(history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[2].set_title("Confusion Matrix")
axes[2].set_ylabel("True Label")
axes[2].set_xlabel("Predicted Label")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage1_results.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlots saved to {RESULTS_DIR}/stage1_results.png")

print("\n" + "=" * 60)
print("STAGE 1 COMPLETE")
print(f"Best model: {RESULTS_DIR}/vgg16_best.keras")
print("=" * 60)
