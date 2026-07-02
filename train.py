#!/usr/bin/env python
# coding: utf-8
"""
Brain Tumor MRI Classification using VGG16 Transfer Learning
==============================================================
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEED = 42
IMG_SIZE = 224


def set_seeds(seed: int = SEED) -> None:
    """Make runs reproducible across numpy / tensorflow / python's random."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def download_from_kaggle(dataset_name: str, destination_dir: Path) -> Path:
    """Download a dataset from Kaggle and extract it."""
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading dataset from Kaggle: %s", dataset_name)

    if shutil.which("kaggle"):
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(destination_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        try:
            import kagglehub

            return Path(kagglehub.dataset_download(dataset_name))
        except ImportError as exc:
            raise ImportError(
                "Kaggle CLI or kagglehub not found. Install: pip install kaggle kagglehub"
            ) from exc

    for file in destination_dir.iterdir():
        if file.suffix == ".zip":
            log.info("Extracting %s...", file.name)
            with zipfile.ZipFile(file, "r") as archive:
                archive.extractall(destination_dir)
            file.unlink(missing_ok=True)
            break

    return destination_dir


def load_dataset(dataset_path: Path, img_size: int = IMG_SIZE):
    """Load images + labels from a folder-per-class directory structure."""
    image_paths = list(paths.list_images(str(dataset_path)))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {dataset_path}")

    images, labels = [], []
    skipped = 0

    for i, image_path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            log.info("Loaded %d/%d images...", i + 1, len(image_paths))

        # pathlib instead of os.path.sep splitting -> works on every OS
        label = Path(image_path).parent.name

        image = cv2.imread(image_path)
        if image is None:
            skipped += 1
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size))
        images.append(image)
        labels.append(label)

    if skipped:
        log.warning("Skipped %d unreadable images.", skipped)

    log.info("Total images loaded: %d", len(images))
    return np.array(images, dtype="float32") / 255.0, np.array(labels)


def encode_labels(labels: np.ndarray):
    """
    LabelEncoder -> integer indices -> to_categorical -> one-hot.
    (The original LabelBinarizer + to_categorical combo only worked by
    coincidence for exactly 2 classes; this generalizes correctly.)
    """
    encoder = LabelEncoder()
    integer_labels = encoder.fit_transform(labels)
    one_hot = to_categorical(integer_labels)
    return one_hot, encoder


def build_model(num_classes: int, img_size: int = IMG_SIZE, learning_rate: float = 1e-3):
    """Build a VGG16-based transfer learning model with a frozen base."""
    base_model = VGG16(
        weights="imagenet",
        input_tensor=Input(shape=(img_size, img_size, 3)),
        include_top=False,
    )

    x = base_model.output
    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten(name="flatten")(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_last_block(base_model, num_layers: int = 4) -> None:
    """Unfreeze the last few conv layers of VGG16 for fine-tuning."""
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True


def get_callbacks(output_dir: Path, prefix: str, history_csv_path: Path, csv_append: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            str(output_dir / f"checkpoint_{prefix}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        # Both phases append to the SAME training_history.csv so the repo
        # ends up with one continuous record instead of two separate files.
        CSVLogger(str(history_csv_path), append=csv_append),
    ]


def plot_history(histories, output_path: Path) -> None:
    """Plot loss and accuracy in separate, readable subplots across all
    training phases (head training + fine-tuning) concatenated together."""
    loss, val_loss, acc, val_acc = [], [], [], []
    for h in histories:
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]

    epochs_range = np.arange(1, len(loss) + 1)
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, loss, label="train_loss")
    axes[0].plot(epochs_range, val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, acc, label="train_acc")
    axes[1].plot(epochs_range, val_acc, label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle("Training Loss and Accuracy on Brain Tumor Dataset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    log.info("Training plot saved to %s", output_path)


def plot_confusion_matrix(cm: np.ndarray, class_names, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    log.info("Confusion matrix plot saved to %s", output_path)


def plot_roc_curve(y_true_onehot: np.ndarray, y_pred_proba: np.ndarray, class_names, output_path: Path) -> None:
    """
    Plot ROC curve(s) with AUC.
    - Binary (2 classes): a single curve for the positive class.
    - Multiclass (3+): one-vs-rest curve per class.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    num_classes = len(class_names)

    if num_classes == 2:
        # Use class index 1 as "positive" (LabelEncoder sorts alphabetically,
        # e.g. ['no', 'yes'] -> 'yes' = index 1).
        fpr, tpr, _ = roc_curve(y_true_onehot[:, 1], y_pred_proba[:, 1])
        auc_score = roc_auc_score(y_true_onehot[:, 1], y_pred_proba[:, 1])
        ax.plot(fpr, tpr, label=f"{class_names[1]} (AUC = {auc_score:.3f})")
    else:
        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            auc_score = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    log.info("ROC curve saved to %s", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VGG16-based brain tumor classifier.")
    parser.add_argument("--dataset-path", type=str, default="brain_tumor_dataset")
    parser.add_argument("--dataset-name", type=str,
                         default="navoneel/brain-mri-images-for-brain-tumor-detection")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-epochs", type=int, default=15,
                         help="Epochs for training the classification head (base frozen).")
    parser.add_argument("--finetune-epochs", type=int, default=10,
                         help="Epochs for fine-tuning the last VGG16 block.")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    # parse_known_args (not parse_args) so this works unmodified in
    # Jupyter/Colab, which injects its own "-f <kernel.json>" argument
    # that argparse would otherwise choke on.
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    set_seeds()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        log.info("Dataset not found at %s. Downloading from Kaggle...", dataset_path)
        download_from_kaggle(args.dataset_name, dataset_path)
    else:
        log.info("Using existing dataset at %s", dataset_path)

    # --- Load & preprocess ---
    images, raw_labels = load_dataset(dataset_path, img_size=args.img_size)
    one_hot_labels, encoder = encode_labels(raw_labels)
    class_names = list(encoder.classes_)
    log.info("Classes: %s | Dataset shape: %s", class_names, images.shape)

    with open(output_dir / "label_classes.json", "w") as f:
        json.dump(class_names, f)

    # --- Train / val / test split (no leakage: test set is untouched until final eval) ---
    train_X, temp_X, train_Y, temp_Y = train_test_split(
        images, one_hot_labels,
        test_size=(args.val_size + args.test_size),
        random_state=SEED,
        stratify=one_hot_labels,
    )
    relative_test_size = args.test_size / (args.val_size + args.test_size)
    val_X, test_X, val_Y, test_Y = train_test_split(
        temp_X, temp_Y,
        test_size=relative_test_size,
        random_state=SEED,
        stratify=temp_Y,
    )
    log.info("Train/Val/Test sizes: %d / %d / %d", len(train_X), len(val_X), len(test_X))

    # --- Class weights (handles tumor/no-tumor imbalance) ---
    integer_train_labels = np.argmax(train_Y, axis=1)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(integer_train_labels),
        y=integer_train_labels,
    )
    class_weights = dict(enumerate(class_weight_values))
    log.info("Class weights: %s", class_weights)

    # --- Data augmentation ---
    train_generator = ImageDataGenerator(
        fill_mode="nearest",
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    # --- Build model ---
    model, base_model = build_model(
        num_classes=len(class_names), img_size=args.img_size, learning_rate=args.head_lr
    )
    model.summary(print_fn=log.info)

    batch_size = args.batch_size
    # NOTE: no manual steps_per_epoch/validation_steps here. flow() is a
    # batched generator, so Keras infers the correct step count per epoch
    # on its own; a hand-computed len(train_X) // batch_size can silently
    # truncate the last partial batch and under-use the data each epoch.
    train_data = train_generator.flow(train_X, train_Y, batch_size=batch_size)
    history_csv_path = output_dir / "training_history.csv"

    # --- Phase 1: train the head with VGG16 frozen ---
    log.info("Phase 1: training classification head (base frozen)...")
    history_head = model.fit(
        train_data,
        validation_data=(val_X, val_Y),
        epochs=args.head_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(output_dir, "head", history_csv_path, csv_append=False),
    )

    # --- Phase 2: fine-tune the last VGG16 conv block at a low LR ---
    log.info("Phase 2: fine-tuning last VGG16 block...")
    unfreeze_last_block(base_model, num_layers=4)
    model.compile(
        optimizer=Adam(learning_rate=args.finetune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_finetune = model.fit(
        train_data,
        validation_data=(val_X, val_Y),
        epochs=args.finetune_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(output_dir, "finetune", history_csv_path, csv_append=True),
    )

    # --- Save model ---
    model.save(output_dir / "brain_tumor_detector.keras")
    log.info("Model saved to %s", output_dir / "brain_tumor_detector.keras")

    # --- Evaluate on the held-out test set (never seen during training/val) ---
    log.info("Evaluating on held-out test set...")
    predictions = model.predict(test_X, batch_size=batch_size)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(test_Y, axis=1)

    report = classification_report(actual_classes, predicted_classes, target_names=class_names)
    log.info("Classification Report:\n%s", report)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(actual_classes, predicted_classes)
    log.info("Confusion Matrix:\n%s", cm)
    accuracy = np.trace(cm) / np.sum(cm)
    log.info("Test Accuracy: %.4f", accuracy)

    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    plot_roc_curve(test_Y, predictions, class_names, output_dir / "roc_curve.png")
    plot_history([history_head, history_finetune], output_dir / "plot.jpg")

    log.info("Done. All artifacts saved to %s", output_dir)


if __name__ == "__main__":
    main()
