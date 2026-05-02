"""
Stage 4: Binary ASD vs. TD Classification with CBAM Attention
Dataset: FADC Dataset (Face ASD Dataset for Children)
Task: Binary ASD vs. Typically Developing classification
Author: Aliza Raza | Student ID: 250915589
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Dense, Dropout, Multiply, Reshape, Add, Lambda, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET_PATH = "/Users/alizaraza/Downloads/FADC-Dataset-main/FADC DATASET"
SPLIT_PATH   = os.path.expanduser("~/Desktop/MSc_Project/FADC_split")
RESULTS_DIR  = os.path.expanduser("~/Desktop/MSc_Project/results_stage4")
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 50
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("STAGE 4: BINARY ASD vs. TD — CBAM ATTENTION CNN")
print("Dataset: FADC Dataset")
print(f"Results: {RESULTS_DIR}")
print("=" * 60)

# ── CREATE TRAIN/VAL/TEST SPLIT ───────────────────────────────────────────────
def create_split(src_dir, split_dir, test_size=0.2, val_size=0.1, seed=42):
    """Split flat ASD/ and TD/ folders into train/val/test structure."""
    if os.path.exists(split_dir):
        print(f"Split already exists at {split_dir}, skipping.")
        return
    print("\nCreating 70/10/20 train/val/test split...")
    for cls in ["ASD", "TD"]:
        src = os.path.join(src_dir, cls)
        files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        train_val, test = train_test_split(files, test_size=test_size, random_state=seed)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=seed)
        for split, subset in [("train", train), ("val", val), ("test", test)]:
            dst = os.path.join(split_dir, split, cls)
            os.makedirs(dst, exist_ok=True)
            for f in subset:
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        print(f"  {cls}: {len(train)} train | {len(val)} val | {len(test)} test")
    print("Split complete.")

create_split(DATASET_PATH, SPLIT_PATH)

# ── DATA GENERATORS ───────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(SPLIT_PATH, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=True, seed=42
)
val_gen = val_test_datagen.flow_from_directory(
    os.path.join(SPLIT_PATH, "val"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    os.path.join(SPLIT_PATH, "test"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)

CLASS_NAMES = list(train_gen.class_indices.keys())
print(f"\nClasses: {train_gen.class_indices}")
print(f"Training: {train_gen.samples} | Validation: {val_gen.samples} | Test: {test_gen.samples}")

# ── CBAM MODULES ──────────────────────────────────────────────────────────────
def channel_attention(x, ratio=8):
    channels = x.shape[-1]
    avg = GlobalAveragePooling2D()(x)
    mx  = GlobalMaxPooling2D()(x)
    dense1 = Dense(channels // ratio, activation="relu")
    dense2 = Dense(channels, activation="sigmoid")
    avg_out = dense2(dense1(avg))
    max_out = dense2(dense1(mx))
    scale = Add()([avg_out, max_out])
    scale = Reshape((1, 1, channels))(scale)
    return Multiply()([x, scale])

def spatial_attention(x, kernel_size=7):
    avg = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    mx  = Lambda(lambda t: tf.reduce_max(t,  axis=-1, keepdims=True))(x)
    concat = Concatenate(axis=-1)([avg, mx])
    attn = Conv2D(1, kernel_size, padding="same", activation="sigmoid")(concat)
    return Multiply()([x, attn])

def cbam_block(x, ratio=8, kernel_size=7):
    x = channel_attention(x, ratio)
    x = spatial_attention(x, kernel_size)
    return x

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
print("\nBuilding VGG16 + CBAM model for binary classification...")
base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Unfreeze blocks 4 and 5
for layer in base.layers:
    layer.trainable = "block4" in layer.name or "block5" in layer.name

# Apply CBAM at blocks 4 and 5
block4_out = base.get_layer("block4_pool").output
block5_out = base.get_layer("block5_conv3").output

attn4 = cbam_block(block4_out, ratio=8)
attn5 = cbam_block(block5_out, ratio=16)

x = MaxPooling2D(pool_size=(2, 2))(attn5)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)  # Binary output

model = Model(inputs=base.input, outputs=output, name="VGG16_CBAM_Binary")
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
print(f"Total parameters:     {model.count_params():,}")

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=12,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        os.path.join(RESULTS_DIR, "stage4_best.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-7, verbose=1)
]

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("\n--- Training VGG16 + CBAM Binary Classifier ---")
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
y_pred_prob = model.predict(test_gen, verbose=1).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = test_gen.classes

accuracy = np.mean(y_pred == y_true)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

print(f"\nTest Accuracy:  {accuracy*100:.2f}%")
print(f"Sensitivity:    {sensitivity*100:.2f}%")
print(f"Specificity:    {specificity*100:.2f}%")
print(f"AUC:            {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ── SAVE ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(RESULTS_DIR, "history.npy"),      history.history)
np.save(os.path.join(RESULTS_DIR, "y_true.npy"),       y_true)
np.save(os.path.join(RESULTS_DIR, "y_pred_prob.npy"),  y_pred_prob)

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
fig.suptitle(f"Stage 4 — Binary ASD vs. TD | Accuracy: {accuracy*100:.1f}% | AUC: {roc_auc:.2f}", fontsize=13)

# Accuracy
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True)

# Loss
axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[2].set_title("Confusion Matrix")
axes[2].set_ylabel("True"); axes[2].set_xlabel("Predicted")

# ROC Curve
axes[3].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.2f})")
axes[3].plot([0,1],[0,1], color="navy", lw=1, linestyle="--")
axes[3].set_title("ROC Curve")
axes[3].set_xlabel("False Positive Rate")
axes[3].set_ylabel("True Positive Rate")
axes[3].legend(); axes[3].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage4_results.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── ATTENTION HEATMAPS ────────────────────────────────────────────────────────
print("\nGenerating attention heatmaps...")
feat_model = Model(inputs=model.input, outputs=model.get_layer("block5_conv3").output)
heatmap_dir = os.path.join(RESULTS_DIR, "heatmaps")
os.makedirs(heatmap_dir, exist_ok=True)

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
fig2.suptitle("Stage 4 — CBAM Attention Heatmaps: ASD vs. TD", fontsize=14)

for idx, cls in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(SPLIT_PATH, "test", cls)
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not files: continue
    img_path = os.path.join(cls_dir, files[0])
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    features = feat_model.predict(img_array[np.newaxis,...], verbose=0)
    attn_map = np.mean(features[0], axis=-1)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_resized = tf.image.resize(attn_map[..., np.newaxis], IMG_SIZE).numpy()[..., 0]
    pred = model.predict(img_array[np.newaxis,...], verbose=0)[0][0]
    pred_label = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
    conf = pred*100 if pred > 0.5 else (1-pred)*100

    axes2[0, idx].imshow(img_array)
    axes2[0, idx].set_title(f"Original\nTrue: {cls}", fontsize=10)
    axes2[0, idx].axis("off")
    axes2[1, idx].imshow(img_array)
    axes2[1, idx].imshow(attn_resized, cmap="jet", alpha=0.45)
    axes2[1, idx].set_title(f"Attention Overlay\nPred: {pred_label} ({conf:.1f}%)", fontsize=10)
    axes2[1, idx].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stage4_heatmaps.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FULL PIPELINE SUMMARY")
print(f"  Stage 1 — VGG16 Baseline (emotion):        42.04%")
print(f"  Stage 2 — VGG16+Xception Ensemble (emotion): 33.63%")
print(f"  Stage 3 — VGG16+CBAM Attention (emotion):  50.00%")
print(f"  Stage 4 — VGG16+CBAM Binary ASD vs. TD:    {accuracy*100:.2f}%")
print(f"\n  AUC:         {roc_auc:.4f}")
print(f"  Sensitivity: {sensitivity*100:.2f}%")
print(f"  Specificity: {specificity*100:.2f}%")
print(f"  Results:     {RESULTS_DIR}")
print("=" * 60)