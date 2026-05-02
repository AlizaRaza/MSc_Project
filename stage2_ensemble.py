"""
Stage 2: VGG16 + Xception Ensemble - Emotion Classification
Dataset: Autism Emotion Recognition Dataset (FER-Autism)
Task: 6-class facial emotion classification with soft-voting ensemble
Author: Aliza Raza | Student ID: 250915589
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import VGG16, Xception
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
RESULTS_DIR   = os.path.expanduser("~/Desktop/MSc_Project/results_stage2")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("STAGE 2: VGG16 + XCEPTION ENSEMBLE — Emotion Classification")
print("Dataset: FER-Autism")
print(f"Results: {RESULTS_DIR}")
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

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training", shuffle=True, seed=42
)
val_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation", shuffle=False, seed=42
)
test_gen = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

NUM_CLASSES = len(train_gen.class_indices)
CLASS_NAMES = list(train_gen.class_indices.keys())
print(f"\nClasses ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"Training: {train_gen.samples} | Validation: {val_gen.samples} | Test: {test_gen.samples}")

# ── BUILD MODEL FUNCTION ───────────────────────────────────────────────────────
def build_model(base, name, lr=1e-4):
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out, name=name)
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def get_callbacks(name):
    return [
        EarlyStopping(monitor="val_accuracy", patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(RESULTS_DIR, f"{name}_best.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1)
    ]

def fine_tune_and_train(model, base, block_name, train_gen, val_gen, lr=1e-5):
    for layer in base.layers:
        if block_name in layer.name:
            layer.trainable = True
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ── TRAIN VGG16 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING MODEL 1: VGG16")
print("=" * 60)

vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
vgg_model = build_model(vgg_base, "vgg16")
print(f"VGG16 parameters: {vgg_model.count_params():,}")

print("\n--- VGG16 Phase 1: Head only ---")
h1_vgg = vgg_model.fit(train_gen, validation_data=val_gen,
                        epochs=20, callbacks=get_callbacks("vgg16"), verbose=1)

print("\n--- VGG16 Phase 2: Fine-tuning block5 ---")
vgg_model = fine_tune_and_train(vgg_model, vgg_base, "block5", train_gen, val_gen)
h2_vgg = vgg_model.fit(train_gen, validation_data=val_gen,
                        epochs=EPOCHS, callbacks=get_callbacks("vgg16"), verbose=1)

# ── TRAIN XCEPTION ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING MODEL 2: XCEPTION")
print("=" * 60)

xc_base = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
xc_model = build_model(xc_base, "xception")
print(f"Xception parameters: {xc_model.count_params():,}")

print("\n--- Xception Phase 1: Head only ---")
h1_xc = xc_model.fit(train_gen, validation_data=val_gen,
                      epochs=20, callbacks=get_callbacks("xception"), verbose=1)

print("\n--- Xception Phase 2: Fine-tuning last block ---")
xc_model = fine_tune_and_train(xc_model, xc_base, "block14", train_gen, val_gen)
h2_xc = xc_model.fit(train_gen, validation_data=val_gen,
                      epochs=EPOCHS, callbacks=get_callbacks("xception"), verbose=1)

# ── INDIVIDUAL EVALUATION ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("INDIVIDUAL MODEL EVALUATION")
print("=" * 60)

test_gen.reset()
vgg_preds = vgg_model.predict(test_gen, verbose=1)
test_gen.reset()
xc_preds = xc_model.predict(test_gen, verbose=1)

y_true = test_gen.classes

vgg_acc = np.mean(np.argmax(vgg_preds, axis=1) == y_true)
xc_acc  = np.mean(np.argmax(xc_preds,  axis=1) == y_true)
print(f"\nVGG16 test accuracy:   {vgg_acc*100:.2f}%")
print(f"Xception test accuracy: {xc_acc*100:.2f}%")

# ── SOFT VOTING ENSEMBLE ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SOFT VOTING ENSEMBLE")
print("=" * 60)

# Equal weight soft voting
ensemble_preds = (vgg_preds + xc_preds) / 2.0
y_ensemble = np.argmax(ensemble_preds, axis=1)
ens_acc = np.mean(y_ensemble == y_true)
print(f"Ensemble test accuracy: {ens_acc*100:.2f}%")

print("\nEnsemble Classification Report:")
print(classification_report(y_true, y_ensemble, target_names=CLASS_NAMES))

# ── SAVE ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(RESULTS_DIR, "vgg_preds.npy"),      vgg_preds)
np.save(os.path.join(RESULTS_DIR, "xc_preds.npy"),       xc_preds)
np.save(os.path.join(RESULTS_DIR, "ensemble_preds.npy"), ensemble_preds)
np.save(os.path.join(RESULTS_DIR, "y_true.npy"),         y_true)

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle(f"Stage 2 — VGG16+Xception Ensemble | Accuracy: {ens_acc*100:.1f}%", fontsize=13)

# Combine histories
def combine(h1, h2, key):
    return h1.history[key] + h2.history[key]

# Accuracy comparison
axes[0].plot(combine(h1_vgg, h2_vgg, "val_accuracy"), label=f"VGG16 val ({vgg_acc*100:.1f}%)")
axes[0].plot(combine(h1_xc,  h2_xc,  "val_accuracy"), label=f"Xception val ({xc_acc*100:.1f}%)")
axes[0].axhline(ens_acc, color='red', linestyle='--', label=f"Ensemble ({ens_acc*100:.1f}%)")
axes[0].set_title("Validation Accuracy Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# Loss comparison
axes[1].plot(combine(h1_vgg, h2_vgg, "val_loss"), label="VGG16 val loss")
axes[1].plot(combine(h1_xc,  h2_xc,  "val_loss"), label="Xception val loss")
axes[1].set_title("Validation Loss Comparison")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

# Confusion Matrix — Ensemble
cm = confusion_matrix(y_true, y_ensemble)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[2].set_title("Ensemble Confusion Matrix")
axes[2].set_ylabel("True Label")
axes[2].set_xlabel("Predicted Label")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage2_results.png"), dpi=150, bbox_inches='tight')
plt.show()

# ── ACCURACY SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2 SUMMARY")
print(f"  Stage 1 VGG16 baseline: 42.04%")
print(f"  Stage 2 VGG16:          {vgg_acc*100:.2f}%")
print(f"  Stage 2 Xception:       {xc_acc*100:.2f}%")
print(f"  Stage 2 Ensemble:       {ens_acc*100:.2f}%")
print(f"  Results saved to:       {RESULTS_DIR}")
print("=" * 60)
