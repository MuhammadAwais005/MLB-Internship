"""Train and evaluate a convolutional classifier for Fashion-MNIST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras import layers, models


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"


def build_model() -> tf.keras.Model:
    """Create the convolutional image classifier."""
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ],
        name="fashion_mnist_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    """Save accuracy and loss curves for training and validation."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history.history["accuracy"]) + 1)

    axes[0].plot(epochs, history.history["accuracy"], marker="o", label="Training")
    axes[0].plot(epochs, history.history["val_accuracy"], marker="o", label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.history["loss"], marker="o", label="Training")
    axes[1].plot(epochs, history.history["val_loss"], marker="o", label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.suptitle("Fashion-MNIST CNN Training")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, output_dir: Path
) -> np.ndarray:
    """Save the test-set confusion matrix and return its counts."""
    matrix = confusion_matrix(
        labels, predictions, labels=np.arange(len(CLASS_NAMES))
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    display.plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )
    axis.set_title("Fashion-MNIST Test Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return matrix


def save_sample_predictions(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> None:
    """Save a labeled grid of test images and model predictions."""
    sample_count = 15
    figure, axes = plt.subplots(3, 5, figsize=(13, 8))
    for index, axis in enumerate(axes.flat):
        actual = int(labels[index])
        predicted = int(predictions[index])
        color = "green" if actual == predicted else "red"
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(
            f"Pred: {CLASS_NAMES[predicted]}\n"
            f"True: {CLASS_NAMES[actual]} "
            f"({probabilities[index, predicted]:.0%})",
            color=color,
            fontsize=8,
        )
        axis.axis("off")

    figure.suptitle(f"Sample Test Predictions (first {sample_count} images)")
    figure.tight_layout()
    figure.savefig(output_dir / "sample_predictions.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def run(epochs: int, batch_size: int, output_dir: Path) -> dict[str, object]:
    """Train, evaluate, and save all project artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    train_images = train_images.astype("float32")[..., np.newaxis] / 255.0
    test_images = test_images.astype("float32")[..., np.newaxis] / 255.0

    model = build_model()
    model.summary()
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )

    train_loss, train_accuracy = model.evaluate(
        train_images, train_labels, batch_size=batch_size, verbose=0
    )
    test_loss, test_accuracy = model.evaluate(
        test_images, test_labels, batch_size=batch_size, verbose=0
    )
    probabilities = model.predict(test_images, batch_size=batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    model.save(output_dir / "fashion_mnist_cnn.keras")
    save_training_curves(history, output_dir)
    matrix = save_confusion_matrix(test_labels, predictions, output_dir)
    save_sample_predictions(
        test_images, test_labels, predictions, probabilities, output_dir
    )

    metrics: dict[str, object] = {
        "dataset": "Fashion-MNIST",
        "training_images": int(train_images.shape[0]),
        "test_images": int(test_images.shape[0]),
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_split": 0.1,
        "architecture": [
            "Conv2D(32, 3x3, relu)",
            "MaxPooling2D(2x2)",
            "Conv2D(64, 3x3, relu)",
            "MaxPooling2D(2x2)",
            "Flatten",
            "Dense(128, relu)",
            "Dropout(0.3)",
            "Dense(10, softmax)",
        ],
        "training_accuracy": float(train_accuracy),
        "training_loss": float(train_loss),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "final_epoch_training_accuracy": float(history.history["accuracy"][-1]),
        "final_epoch_validation_accuracy": float(
            history.history["val_accuracy"][-1]
        ),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("\nEvaluation on complete datasets")
    print(f"Training accuracy: {train_accuracy:.2%} (loss {train_loss:.4f})")
    print(f"Test accuracy:     {test_accuracy:.2%} (loss {test_loss:.4f})")
    print(f"Artifacts saved to: {output_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Fashion-MNIST CNN."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training and evaluation batch size."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the model, metrics, and figures.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.epochs, arguments.batch_size, arguments.output_dir)
