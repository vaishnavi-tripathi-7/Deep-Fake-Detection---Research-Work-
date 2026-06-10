"""
================================================================================
GAN-Augmented XceptionNet Training for Deepfake Detection
================================================================================

PURPOSE
-------
This script is the NEXT STEP after running:
  1. dcgan_deepfake_augmentation.py   ← generates 5000 synthetic fake images
  2. Manual filtering of generated_gan_fakes/

It trains a new XceptionNet variant ("GAN-Augmented XceptionNet") on:
  - Original train/fake/ images
  - Manually-filtered GAN-generated fakes (merged into train/fake/)

Then evaluates all FOUR models side-by-side:
  ① XceptionNet Baseline              (hist_xception, model_xception_ft)
  ② EfficientNet Baseline             (hist_effnet,   model_effnet_ft)
  ③ Traditional Aug XceptionNet       (hist_aug,      model_aug)
  ④ GAN-Augmented XceptionNet  ← NEW  (hist_gan_aug,  model_gan_aug)

ASSUMPTION
----------
You have already run the baseline notebook (notebook1f31edab98.ipynb) in the
same Colab session, so the following variables are already in scope:
  TRAIN_DIR, VAL_DIR, TEST_DIR, STAGED, device
  BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, SEED
  test_pytorch_dataset, test_loader_aug
  hist_xception, hist_effnet, hist_aug
  all_labels_xception, all_preds_xception, fpr_xception, tpr_xception, roc_auc_xception
  all_labels_effnet,   all_preds_effnet,   fpr_effnet,   tpr_effnet,   roc_auc_effnet
  all_labels_aug,      all_preds_aug,      fpr_aug,      tpr_aug,      roc_auc_aug
  accuracy_xception, precision_xception, recall_xception, f1_xception
  accuracy_effnet,   precision_effnet,   recall_effnet,   f1_effnet
  accuracy_aug,      precision_aug,      recall_aug,      f1_aug
  train_model   ← the utility function from the baseline notebook

If you are running this script STANDALONE (not in a shared Colab session),
set STANDALONE_MODE = True below — it will re-create minimal stubs so the
script still runs without crashing.
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Standalone / Shared-Session Switch
# ─────────────────────────────────────────────────────────────────────────────

STANDALONE_MODE = False   # Set True only if running outside the baseline session

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Imports
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
import random
import time
import copy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Resolve Shared Session Variables (or create stubs)
# ─────────────────────────────────────────────────────────────────────────────

if STANDALONE_MODE:
    # ── Minimal stubs so this file is importable standalone ────────────────
    SEED       = 42
    IMG_HEIGHT = 224
    IMG_WIDTH  = 224
    BATCH_SIZE = 32
    STAGED     = "staged_dataset"          # must already exist on disk
    TRAIN_DIR  = os.path.join(STAGED, "train")
    VAL_DIR    = os.path.join(STAGED, "val")
    TEST_DIR   = os.path.join(STAGED, "test")
    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Placeholder history dicts — filled with zeros so comparison plots render
    def _empty_hist(n=10):
        return {"train_loss": [0.5]*n, "val_loss": [0.5]*n,
                "train_acc":  [0.5]*n, "val_acc":  [0.5]*n}
    hist_xception = _empty_hist()
    hist_effnet   = _empty_hist()
    hist_aug      = _empty_hist()

    # Placeholder metrics
    accuracy_xception  = precision_xception = recall_xception = f1_xception  = 0.0
    accuracy_effnet    = precision_effnet   = recall_effnet   = f1_effnet    = 0.0
    accuracy_aug       = precision_aug      = recall_aug      = f1_aug       = 0.0
    roc_auc_xception   = roc_auc_effnet     = roc_auc_aug                    = 0.0

    fpr_xception = tpr_xception = np.array([0, 1])
    fpr_effnet   = tpr_effnet   = np.array([0, 1])
    fpr_aug      = tpr_aug      = np.array([0, 1])

    # Stub train_model (returns untrained model immediately)
    def train_model(model, criterion, optimizer, train_loader, val_loader,
                    num_epochs=30, patience=5):
        history = {"train_loss": [], "val_loss": [],
                   "train_acc":  [], "val_acc":  []}
        print("[STANDALONE STUB] train_model called — skipping actual training.")
        return model, history

print(f"[INFO] Device        : {device}")
print(f"[INFO] TRAIN_DIR     : {TRAIN_DIR}")
print(f"[INFO] Standalone    : {STANDALONE_MODE}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: GAN-Generated Images — Merge Into Training Set
# ─────────────────────────────────────────────────────────────────────────────
#
# WORKFLOW:
#   After DCGAN training + manual review, move accepted images from
#   generated_gan_fakes/ into TRAIN_DIR/fake/ so that the augmented
#   dataset is seen by both this script and the existing DataLoaders.
#
# SAFETY:
#   Only TRAIN_DIR/fake/ is modified.
#   VAL_DIR and TEST_DIR are NEVER touched.
# ─────────────────────────────────────────────────────────────────────────────

GAN_GENERATED_DIR    = "generated_gan_fakes"       # output of DCGAN script
GAN_FILTERED_DIR     = "gan_filtered_fakes"        # optional: post manual review
TRAIN_FAKE_DIR       = os.path.join(TRAIN_DIR, "fake")
BACKUP_TRAIN_FAKE    = os.path.join(TRAIN_DIR, "fake_original_backup")

def merge_gan_images_into_train(
    src_dir: str,
    dst_dir: str,
    backup_dir: str,
    prefix: str = "gan_",
) -> int:
    """
    Copy GAN-generated images from src_dir into dst_dir.

    Parameters
    ----------
    src_dir    : Folder containing GAN-generated (or filtered) PNG/JPG images.
    dst_dir    : train/fake/ — images are ADDED here, nothing is deleted.
    backup_dir : A one-time backup of the original train/fake/ contents.
    prefix     : Prefix added to every copied filename to avoid collisions.

    Returns
    -------
    Number of images successfully copied.
    """
    if not os.path.isdir(src_dir):
        print(f"[WARN] GAN source directory not found: {src_dir}")
        print("       Skipping merge — GAN images will NOT be added.")
        return 0

    # ── One-time backup of original train/fake/ ────────────────────────────
    if not os.path.exists(backup_dir):
        print(f"[INFO] Creating backup of original train/fake/ → {backup_dir}")
        shutil.copytree(dst_dir, backup_dir)
        print(f"[INFO] Backup done ({len(os.listdir(backup_dir))} files).")
    else:
        print(f"[INFO] Backup already exists at {backup_dir} — skipping.")

    # ── Copy GAN images ────────────────────────────────────────────────────
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    copied = 0
    for fname in sorted(os.listdir(src_dir)):
        if os.path.splitext(fname)[1].lower() not in valid_exts:
            continue
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, prefix + fname)
        if not os.path.exists(dst_path):               # don't overwrite
            shutil.copy2(src_path, dst_path)
            copied += 1

    print(f"[INFO] Merged {copied} GAN images from {src_dir} → {dst_dir}")
    print(f"[INFO] train/fake/ now contains {len(os.listdir(dst_dir))} images total.")
    return copied


# Decide which source directory to use
# (filtered > raw if filtered folder exists and is non-empty)
if (os.path.isdir(GAN_FILTERED_DIR) and
        len(os.listdir(GAN_FILTERED_DIR)) > 0):
    gan_source = GAN_FILTERED_DIR
    print(f"[INFO] Using FILTERED GAN images from: {gan_source}")
else:
    gan_source = GAN_GENERATED_DIR
    print(f"[INFO] Using RAW GAN images from     : {gan_source}")
    print("       (If you have a filtered folder, name it 'gan_filtered_fakes/')")

n_merged = merge_gan_images_into_train(
    src_dir    = gan_source,
    dst_dir    = TRAIN_FAKE_DIR,
    backup_dir = BACKUP_TRAIN_FAKE,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Transforms for GAN-Augmented Model
# ─────────────────────────────────────────────────────────────────────────────
#
# These are IDENTICAL to the augmented-model transforms in the baseline so
# that the only difference between `model_aug` and `model_gan_aug` is the
# training data (real-only-fakes vs real-fakes + GAN-fakes).
#
# Note on normalisation:
#   DCGAN output was [-1,1] (Tanh).  We denormalised to [0,1] PIL images
#   when saving in dcgan_deepfake_augmentation.py — so standard ImageNet
#   normalisation applies here exactly as before.
# ─────────────────────────────────────────────────────────────────────────────

class AddGaussianNoise:
    """Randomly add Gaussian noise to a tensor (applied after ToTensor)."""
    def __init__(self, mean: float = 0.0, std: float = 0.01, p: float = 0.1):
        self.mean, self.std, self.p = mean, std, p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


class SimulateJPEGCompression:
    """Simulate mild JPEG artefacts — helps learn compression-invariant features."""
    def __init__(self, quality_range=(80, 95), p: float = 0.15):
        self.quality_range, self.p = quality_range, p

    def __call__(self, tensor):
        if random.random() < self.p:
            q = random.randint(*self.quality_range)
            tensor = torch.round(tensor * q) / q
        return tensor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform_gan_aug = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.08, contrast=0.08,
                           saturation=0.05, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
    transforms.ToTensor(),
    AddGaussianNoise(std=0.01, p=0.1),
    SimulateJPEGCompression(quality_range=(80, 95), p=0.15),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform_gan_aug = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print("[INFO] GAN-augmented transforms created.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: DataLoaders for GAN-Augmented Training
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)

train_dataset_gan_aug = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform_gan_aug,
)
val_dataset_gan_aug = datasets.ImageFolder(
    root=VAL_DIR,
    transform=eval_transform_gan_aug,
)
test_dataset_gan_aug = datasets.ImageFolder(
    root=TEST_DIR,
    transform=eval_transform_gan_aug,
)

print(f"[INFO] GAN-aug train size : {len(train_dataset_gan_aug)}")
print(f"       (baseline had      : {len(train_dataset_gan_aug) - n_merged} samples)")
print(f"       GAN images added   : {n_merged}")
print(f"[INFO] Val  size          : {len(val_dataset_gan_aug)}")
print(f"[INFO] Test size          : {len(test_dataset_gan_aug)}")
print(f"[INFO] Class mapping      : {train_dataset_gan_aug.class_to_idx}")

train_loader_gan_aug = DataLoader(
    train_dataset_gan_aug,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)
val_loader_gan_aug = DataLoader(
    val_dataset_gan_aug,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)
test_loader_gan_aug = DataLoader(
    test_dataset_gan_aug,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)

print("[INFO] GAN-augmented DataLoaders ready.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: GAN-Augmented XceptionNet — Architecture
# ─────────────────────────────────────────────────────────────────────────────
#
# IDENTICAL architecture to model_aug (same classifier head, same freeze strategy)
# so that any performance difference is purely due to the richer training data.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[INFO] Initialising GAN-Augmented XceptionNet …")

model_gan_aug = timm.create_model("xception", pretrained=True)

# ── Replace classifier head ────────────────────────────────────────────────
num_ftrs_gan_aug = model_gan_aug.num_features
model_gan_aug.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs_gan_aug, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 2),           # binary: real / fake
)
print("[INFO] Custom 3-layer classifier head added.")

# ── Freeze entire backbone ─────────────────────────────────────────────────
for param in model_gan_aug.parameters():
    param.requires_grad = False
print("[INFO] All backbone layers frozen.")

# ── Unfreeze classifier head ───────────────────────────────────────────────
for param in model_gan_aug.classifier.parameters():
    param.requires_grad = True
print("[INFO] Classifier head unfrozen.")

# ── Unfreeze last 30 backbone parameters (matches baseline) ───────────────
params_list_gan = list(model_gan_aug.parameters())
for param in params_list_gan[-30:]:
    param.requires_grad = True
print("[INFO] Last 30 backbone parameters unfrozen.")

# ── Move to device ─────────────────────────────────────────────────────────
model_gan_aug = model_gan_aug.to(device)
print(f"[INFO] Model moved to: {device}")

# Count trainable params
trainable = sum(p.numel() for p in model_gan_aug.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model_gan_aug.parameters())
print(f"[INFO] Trainable params: {trainable:,} / {total:,}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Loss Function & Optimiser
# ─────────────────────────────────────────────────────────────────────────────

criterion_gan_aug = nn.CrossEntropyLoss()

optimizer_gan_aug = optim.Adam(
    filter(lambda p: p.requires_grad, model_gan_aug.parameters()),
    lr=0.0001,
    betas=(0.9, 0.999),
)

print("[INFO] CrossEntropyLoss + Adam(lr=1e-4) initialised.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Training
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Starting GAN-Augmented XceptionNet Training")
print("=" * 70)

model_gan_aug, hist_gan_aug = train_model(
    model_gan_aug,
    criterion_gan_aug,
    optimizer_gan_aug,
    train_loader_gan_aug,
    val_loader_gan_aug,
    num_epochs=30,
    patience=5,
)

print("\n[✓] GAN-Augmented XceptionNet training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: Training History Plots — GAN-Augmented Model
# ─────────────────────────────────────────────────────────────────────────────

# Safely convert tensor accuracy values to floats (mirrors baseline pattern)
for key in ("train_acc", "val_acc"):
    hist_gan_aug[key] = [
        v.cpu().item() if isinstance(v, torch.Tensor) else float(v)
        for v in hist_gan_aug[key]
    ]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(hist_gan_aug["train_loss"], label="Train Loss",      color="steelblue",  lw=2)
axes[0].plot(hist_gan_aug["val_loss"],   label="Val Loss",        color="darkorange", lw=2)
axes[0].set_title("GAN-Augmented XceptionNet — Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(hist_gan_aug["train_acc"], label="Train Accuracy",   color="steelblue",  lw=2)
axes[1].plot(hist_gan_aug["val_acc"],   label="Val Accuracy",     color="darkorange", lw=2)
axes[1].set_title("GAN-Augmented XceptionNet — Accuracy")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gan_aug_xception_history.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("[INFO] Training history saved → gan_aug_xception_history.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: Evaluation on Test Set
# ─────────────────────────────────────────────────────────────────────────────

model_gan_aug.eval()

all_labels_gan_aug = []
all_preds_gan_aug  = []
all_probs_gan_aug  = []

print("\n[INFO] Evaluating GAN-Augmented XceptionNet on test set …")

with torch.no_grad():
    for inputs, labels in test_loader_gan_aug:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_gan_aug(inputs)
        probs   = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels_gan_aug.extend(labels.cpu().numpy())
        all_preds_gan_aug.extend(preds.cpu().numpy())
        all_probs_gan_aug.extend(probs[:, 1].cpu().numpy())  # P(fake)

all_labels_gan_aug = np.array(all_labels_gan_aug)
all_preds_gan_aug  = np.array(all_preds_gan_aug)
all_probs_gan_aug  = np.array(all_probs_gan_aug)

# ── Metrics ────────────────────────────────────────────────────────────────
accuracy_gan_aug  = accuracy_score (all_labels_gan_aug, all_preds_gan_aug)
precision_gan_aug = precision_score(all_labels_gan_aug, all_preds_gan_aug, average="binary")
recall_gan_aug    = recall_score   (all_labels_gan_aug, all_preds_gan_aug, average="binary")
f1_gan_aug        = f1_score       (all_labels_gan_aug, all_preds_gan_aug, average="binary")
fpr_gan_aug, tpr_gan_aug, _ = roc_curve(all_labels_gan_aug, all_probs_gan_aug)
roc_auc_gan_aug   = auc(fpr_gan_aug, tpr_gan_aug)

print(f"\n  Test Accuracy  (GAN-Aug XceptionNet): {accuracy_gan_aug :.4f}")
print(f"  Test Precision (GAN-Aug XceptionNet): {precision_gan_aug:.4f}")
print(f"  Test Recall    (GAN-Aug XceptionNet): {recall_gan_aug   :.4f}")
print(f"  Test F1-Score  (GAN-Aug XceptionNet): {f1_gan_aug       :.4f}")
print(f"  Test AUC       (GAN-Aug XceptionNet): {roc_auc_gan_aug  :.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: Confusion Matrix — GAN-Augmented Model
# ─────────────────────────────────────────────────────────────────────────────

cm_gan_aug = confusion_matrix(all_labels_gan_aug, all_preds_gan_aug)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm_gan_aug, annot=True, fmt="d", cmap="Blues",
    xticklabels=test_dataset_gan_aug.classes,
    yticklabels=test_dataset_gan_aug.classes,
    ax=ax,
)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — GAN-Augmented XceptionNet (Test Set)")
plt.tight_layout()
plt.savefig("cm_gan_aug_xception.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("[INFO] Confusion matrix saved → cm_gan_aug_xception.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: ROC Curve — GAN-Augmented Model
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_gan_aug, tpr_gan_aug, color="purple", lw=2,
        label=f"GAN-Aug XceptionNet (AUC = {roc_auc_gan_aug:.2f})")
ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Classifier")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — GAN-Augmented XceptionNet")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_gan_aug_xception.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("[INFO] ROC curve saved → roc_gan_aug_xception.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: Save GAN-Augmented Model
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("trained_models", exist_ok=True)
torch.save(model_gan_aug.state_dict(),
           "trained_models/gan_augmented_xception_model.pth")
print("[INFO] Model saved → trained_models/gan_augmented_xception_model.pth")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14: Full 4-Model Comparison — Loss & Accuracy Curves
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Generating 4-Model Comparison Plots")
print("=" * 70)

# ── Safely normalise all history dicts ────────────────────────────────────
def to_float_list(lst):
    return [v.cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for v in lst]

for h in (hist_xception, hist_effnet, hist_aug, hist_gan_aug):
    for k in ("train_acc", "val_acc"):
        h[k] = to_float_list(h[k])

MODEL_STYLES = {
    "XceptionNet Baseline"     : {"hist": hist_xception, "color": "steelblue",   "ls": "-"},
    "EfficientNet Baseline"    : {"hist": hist_effnet,   "color": "darkorange",  "ls": "-"},
    "Aug XceptionNet"          : {"hist": hist_aug,      "color": "seagreen",    "ls": "-"},
    "GAN-Aug XceptionNet"      : {"hist": hist_gan_aug,  "color": "orchid",      "ls": "-"},
}

# ── Loss comparison ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for label, cfg in MODEL_STYLES.items():
    axes[0].plot(cfg["hist"]["train_loss"], color=cfg["color"],
                 linestyle="--", alpha=0.6, label=f"{label} – Train")
    axes[0].plot(cfg["hist"]["val_loss"],   color=cfg["color"],
                 linestyle=cfg["ls"],       label=f"{label} – Val")

axes[0].set_title("Training & Validation Loss — All Models", fontsize=13)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=0.3)

for label, cfg in MODEL_STYLES.items():
    axes[1].plot(cfg["hist"]["train_acc"], color=cfg["color"],
                 linestyle="--", alpha=0.6, label=f"{label} – Train")
    axes[1].plot(cfg["hist"]["val_acc"],   color=cfg["color"],
                 linestyle=cfg["ls"],       label=f"{label} – Val")

axes[1].set_title("Training & Validation Accuracy — All Models", fontsize=13)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(fontsize=8, ncol=2); axes[1].grid(alpha=0.3)

plt.suptitle("Deepfake Detector: 4-Model Training Comparison", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("all_models_loss_acc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("[INFO] Loss/Accuracy comparison saved → all_models_loss_acc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15: Full 4-Model Comparison — ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

ROC_DATA = [
    ("XceptionNet Baseline",  fpr_xception, tpr_xception, roc_auc_xception, "steelblue"),
    ("EfficientNet Baseline", fpr_effnet,   tpr_effnet,   roc_auc_effnet,   "darkorange"),
    ("Aug XceptionNet",       fpr_aug,      tpr_aug,      roc_auc_aug,      "seagreen"),
    ("GAN-Aug XceptionNet",   fpr_gan_aug,  tpr_gan_aug,  roc_auc_gan_aug,  "orchid"),
]

fig, ax = plt.subplots(figsize=(10, 8))
for label, fpr, tpr, roc_auc, color in ROC_DATA:
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{label} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate",  fontsize=13)
ax.set_title("ROC Curve Comparison — All 4 Models", fontsize=15)
ax.legend(loc="lower right", fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("all_models_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("[INFO] ROC comparison saved → all_models_roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16: Full 4-Model Comparison — Metrics Table & Bar Charts
# ─────────────────────────────────────────────────────────────────────────────

metrics_data = {
    "Model": [
        "XceptionNet Baseline",
        "EfficientNet Baseline",
        "Aug XceptionNet",
        "GAN-Aug XceptionNet",
    ],
    "Accuracy" : [accuracy_xception,  accuracy_effnet,  accuracy_aug,  accuracy_gan_aug],
    "Precision": [precision_xception, precision_effnet, precision_aug, precision_gan_aug],
    "Recall"   : [recall_xception,    recall_effnet,    recall_aug,    recall_gan_aug],
    "F1-Score" : [f1_xception,        f1_effnet,        f1_aug,        f1_gan_aug],
    "AUC"      : [roc_auc_xception,   roc_auc_effnet,   roc_auc_aug,   roc_auc_gan_aug],
}

df_metrics = pd.DataFrame(metrics_data)
print("\n" + "─" * 70)
print("  TEST SET PERFORMANCE SUMMARY")
print("─" * 70)
print(df_metrics.to_markdown(index=False, floatfmt=".4f"))
print("─" * 70)

# Save as CSV for later reference
df_metrics.to_csv("model_comparison_metrics.csv", index=False)
print("[INFO] Metrics table saved → model_comparison_metrics.csv")

# ── Bar charts ────────────────────────────────────────────────────────────
METRICS_TO_PLOT = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
COLORS_BAR = ["steelblue", "darkorange", "seagreen", "orchid"]

fig, axes = plt.subplots(1, len(METRICS_TO_PLOT), figsize=(18, 6))

for ax, metric in zip(axes, METRICS_TO_PLOT):
    bars = ax.bar(
        df_metrics["Model"],
        df_metrics[metric],
        color=COLORS_BAR,
        edgecolor="white",
        linewidth=0.8,
    )
    # Annotate bar values
    for bar, val in zip(bars, df_metrics[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    ax.set_title(metric, fontsize=12)
    ax.set_ylim(max(0, df_metrics[metric].min() - 0.05), 1.05)
    ax.set_xticks(range(len(df_metrics["Model"])))
    ax.set_xticklabels(
        ["XceptionNet\nBaseline", "EfficientNet\nBaseline",
         "Aug\nXceptionNet", "GAN-Aug\nXceptionNet"],
        fontsize=8,
    )
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Test Set Metric Comparison — All 4 Models", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("all_models_metrics_bar.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
print("[INFO] Bar chart saved → all_models_metrics_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17: Delta Analysis — How Much Did GAN Augmentation Help?
# ─────────────────────────────────────────────────────────────────────────────
#
# Compare GAN-Aug XceptionNet vs its closest baseline: Aug XceptionNet.
# Same architecture, same traditional augmentation — only the training data
# differs.  Positive Δ = GAN augmentation helped.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  GAN AUGMENTATION IMPACT ANALYSIS")
print("  (GAN-Aug XceptionNet  vs  Traditional Aug XceptionNet)")
print("=" * 70)

deltas = {
    "Accuracy" : accuracy_gan_aug  - accuracy_aug,
    "Precision": precision_gan_aug - precision_aug,
    "Recall"   : recall_gan_aug    - recall_aug,
    "F1-Score" : f1_gan_aug        - f1_aug,
    "AUC"      : roc_auc_gan_aug   - roc_auc_aug,
}

for metric, delta in deltas.items():
    sign  = "+" if delta >= 0 else ""
    arrow = "↑ IMPROVED" if delta > 0.002 else ("↓ REGRESSED" if delta < -0.002 else "≈ SIMILAR")
    print(f"  {metric:<12}: {sign}{delta:+.4f}   {arrow}")

print("=" * 70)
print()
print("  NOTE: Small positive/negative deltas (< 0.002) are within normal")
print("  training variance.  Large positive deltas confirm GAN augmentation")
print("  improved detector generalisation.  Negative deltas on a specific")
print("  metric may indicate the generated images introduced minor noise for")
print("  that metric — review which fake sub-types the GAN learned best.")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18: Load Saved Model Utility (For Future Use / Kaggle Inference)
# ─────────────────────────────────────────────────────────────────────────────

def load_gan_augmented_xception(weights_path: str, device: torch.device):
    """
    Re-create and load the GAN-Augmented XceptionNet model from a saved .pth.

    Usage (e.g. in a Kaggle inference notebook):
    --------------------------------------------
    from gan_augmented_xception_training import load_gan_augmented_xception
    model = load_gan_augmented_xception(
        "trained_models/gan_augmented_xception_model.pth",
        torch.device("cuda")
    )
    """
    model = timm.create_model("xception", pretrained=False)
    num_ftrs = model.num_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded GAN-Aug XceptionNet from {weights_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19: Final Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("GAN-Augmented XceptionNet — Complete")
print("=" * 70)
print()
print("  Files generated this run:")
print("    gan_aug_xception_history.png          ← loss/acc curves (this model)")
print("    cm_gan_aug_xception.png               ← confusion matrix")
print("    roc_gan_aug_xception.png              ← ROC curve")
print("    all_models_loss_acc_curves.png        ← 4-model training curves")
print("    all_models_roc_curves.png             ← 4-model ROC comparison")
print("    all_models_metrics_bar.png            ← 4-model bar chart")
print("    model_comparison_metrics.csv          ← metrics table")
print("    trained_models/")
print("      gan_augmented_xception_model.pth    ← final weights")
print()
print("  Dataset integrity:")
print(f"    train/fake/ : {len(os.listdir(TRAIN_FAKE_DIR))} images (original + GAN)")
print(f"    val/        : UNTOUCHED")
print(f"    test/       : UNTOUCHED")
print()
print("  Next steps:")
print("    1. Compare 4-model metrics — did GAN augmentation raise F1/AUC?")
print("    2. If GAN-Aug underperforms, try stricter manual filtering of GAN images.")
print("    3. Try retraining with only GAN images that look different from FaceForensics++")
print("       artefacts (GAN introduces different frequency patterns vs video-compression")
print("       artefacts in the original fake frames).")
print("=" * 70)
