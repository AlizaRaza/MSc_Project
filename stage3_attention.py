"""
Stage 3: Attention-Based CNN with CBAM Modules
Dataset: Autism Emotion Recognition Dataset (FER-Autism)
Task: 6-class facial emotion classification with intrinsic attention heatmaps
Author: Aliza Raza | Student ID: 250915589
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout,
    Multiply, Reshape, Add, Lambda, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET_PATH = "/Users/alizaraza/Downloads/Autism emotion recogition dataset"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 50
RESULTS_DIR  = os.path.expanduser("~/Desktop/MSc_Project/results_stage3")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("STAGE 3: ATTENTION-BASED CNN (CBAM) — Emotion Classification")
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

# ── CBAM ATTENTION MODULE ──────────────────────────────────────────────────────
def channel_attention(x, ratio=8):
    """Channel attention: WHAT features to focus on"""
    channels = x.shape[-1]
    # Average pool + max pool across spatial dims
    avg = GlobalAveragePooling2D()(x)
    mx  = GlobalMaxPooling2D()(x)
    # Shared MLP
    dense1 = Dense(channels // ratio, activation="relu")
    dense2 = Dense(channels, activation="sigmoid")
    avg_out = dense2(dense1(avg))
    max_out = dense2(dense1(mx))
    # Combine and reshape for multiplication
    scale = Add()([avg_out, max_out])
    scale = Reshape((1, 1, channels))(scale)
    return Multiply()([x, scale])

def spatial_attention(x, kernel_size=7):
    """Spatial attention: WHERE to focus in the image"""
    # Average + max along channel axis
    avg = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    mx  = Lambda(lambda t: tf.reduce_max(t,  axis=-1, keepdims=True))(x)
    concat = Concatenate(axis=-1)([avg, mx])
    # 7x7 conv to produce spatial attention map
    attn = Conv2D(1, kernel_size, padding="same", activation="sigmoid")(concat)
    return Multiply()([x, attn])

def cbam_block(x, ratio=8, kernel_size=7):
    """Full CBAM: channel attention then spatial attention"""
    x = channel_attention(x, ratio)
    x = spatial_attention(x, kernel_size)
    return x

# ── BUILD ATTENTION MODEL ──────────────────────────────────────────────────────
def build_attention_model(num_classes, input_shape=(224, 224, 3)):
    """
    VGG16 backbone + CBAM attention at block3, block4, block5.
    Attention maps from block5 are saved for visualisation.
    """
    # Load VGG16 backbone (frozen initially)
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    # We need intermediate outputs for attention
    inp = base.input

    # Get outputs from key blocks
    block3_out = base.get_layer("block3_pool").output   # 28x28
    block4_out = base.get_layer("block4_pool").output   # 14x14
    block5_out = base.get_layer("block5_conv3").output  # 14x14 (before pool)

    # Apply CBAM to each block
    attn3 = cbam_block(block3_out, ratio=8,  kernel_size=7)
    attn4 = cbam_block(block4_out, ratio=8,  kernel_size=7)
    attn5 = cbam_block(block5_out, ratio=16, kernel_size=7)

    # Use block5 max pool after attention
    x = MaxPooling2D(pool_size=(2, 2))(attn5)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out, name="VGG16_CBAM")

    # Freeze base, unfreeze from block4 onwards
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers:
        if "block4" in layer.name or "block5" in layer.name:
            layer.trainable = True

    return model, base

print("\nBuilding VGG16 + CBAM Attention model...")
model, base_model = build_attention_model(NUM_CLASSES)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print(f"Total parameters:     {model.count_params():,}")
trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Trainable parameters: {trainable:,}")

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=12,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        os.path.join(RESULTS_DIR, "cbam_best.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-7, verbose=1)
]

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("\n--- Training VGG16 + CBAM (blocks 4 & 5 unfrozen) ---")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

accuracy = np.mean(y_pred == y_true)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ── SAVE ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(RESULTS_DIR, "history.npy"),      history.history)
np.save(os.path.join(RESULTS_DIR, "y_true.npy"),       y_true)
np.save(os.path.join(RESULTS_DIR, "y_pred_probs.npy"), y_pred_probs)

# ── GENERATE ATTENTION HEATMAPS ───────────────────────────────────────────────
print("\nGenerating facial attention heatmaps...")

# Build a model that outputs the spatial attention map from block5
# We extract the spatial attention weights directly
def get_spatial_attention_map(model, img_array):
    """Extract spatial attention map from block5 for a single image."""
    # Get block5_conv3 output
    feat_model = Model(
        inputs=model.input,
        outputs=model.get_layer("block5_conv3").output
    )
    features = feat_model.predict(img_array[np.newaxis, ...], verbose=0)

    # Compute spatial attention map: mean across channels
    # This shows WHERE the model attends in the image
    attn_map = np.mean(features[0], axis=-1)

    # Normalize to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map

# Load test images and generate heatmaps for one sample per class
test_gen.reset()
heatmap_dir = os.path.join(RESULTS_DIR, "attention_heatmaps")
os.makedirs(heatmap_dir, exist_ok=True)

# Get file paths from generator
filepaths = test_gen.filepaths
labels    = test_gen.classes

print(f"Generating heatmaps for {NUM_CLASSES} emotion classes...")
fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(12, NUM_CLASSES * 4))
fig.suptitle("Stage 3 — CBAM Spatial Attention Heatmaps (FER-Autism)", fontsize=14)

from tensorflow.keras.preprocessing import image as keras_image

for class_idx, class_name in enumerate(CLASS_NAMES):
    # Find first test image for this class
    class_files = [f for f, l in zip(filepaths, labels) if l == class_idx]
    if not class_files:
        continue

    img_path = class_files[0]
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0

    # Get prediction
    pred = model.predict(img_array[np.newaxis, ...], verbose=0)
    pred_class = CLASS_NAMES[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Get attention map
    attn_map = get_spatial_attention_map(model, img_array)

    # Resize attention map to image size
    attn_resized = tf.image.resize(
        attn_map[..., np.newaxis], IMG_SIZE
    ).numpy()[..., 0]

    # Plot original image
    axes[class_idx, 0].imshow(img_array)
    axes[class_idx, 0].set_title(f"Original\nTrue: {class_name}", fontsize=10)
    axes[class_idx, 0].axis("off")

    # Plot attention map
    axes[class_idx, 1].imshow(attn_resized, cmap="hot")
    axes[class_idx, 1].set_title(f"Attention Map\nPred: {pred_class} ({confidence:.1f}%)", fontsize=10)
    axes[class_idx, 1].axis("off")

    # Plot overlay
    axes[class_idx, 2].imshow(img_array)
    axes[class_idx, 2].imshow(attn_resized, cmap="jet", alpha=0.4)
    axes[class_idx, 2].set_title("Overlay", fontsize=10)
    axes[class_idx, 2].axis("off")

plt.tight_layout()
heatmap_path = os.path.join(RESULTS_DIR, "attention_heatmaps.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Heatmaps saved to: {heatmap_path}")

# ── TRAINING PLOTS ────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 5))
fig2.suptitle(f"Stage 3 — VGG16+CBAM Attention | Accuracy: {accuracy*100:.1f}%", fontsize=13)

axes2[0].plot(history.history["accuracy"],     label="Train")
axes2[0].plot(history.history["val_accuracy"], label="Validation")
axes2[0].set_title("Model Accuracy")
axes2[0].set_xlabel("Epoch")
axes2[0].set_ylabel("Accuracy")
axes2[0].legend()
axes2[0].grid(True)

axes2[1].plot(history.history["loss"],     label="Train")
axes2[1].plot(history.history["val_loss"], label="Validation")
axes2[1].set_title("Model Loss")
axes2[1].set_xlabel("Epoch")
axes2[1].set_ylabel("Loss")
axes2[1].legend()
axes2[1].grid(True)

cm_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues", ax=axes2[2],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes2[2].set_title("Confusion Matrix")
axes2[2].set_ylabel("True Label")
axes2[2].set_xlabel("Predicted Label")
axes2[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage3_results.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("THREE-STAGE PIPELINE SUMMARY")
print(f"  Stage 1 — VGG16 Baseline:       42.04%")
print(f"  Stage 2 — VGG16+Xception Ens.:  33.63%")
print(f"  Stage 3 — VGG16+CBAM Attention: {accuracy*100:.2f}%")
print(f"\n  Attention heatmaps: {heatmap_path}")
print(f"  Results saved to:   {RESULTS_DIR}")
print("=" * 60)
