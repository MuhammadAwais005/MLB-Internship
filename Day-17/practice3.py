
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# 2. Normalize the images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 3. Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# 4. Define class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 5. Load the trained model (no retraining)
model = tf.keras.models.load_model("fashion_cnn.keras")

print("Trained CNN model loaded successfully!")

# 6. Evaluate training accuracy and loss
train_loss, train_accuracy = model.evaluate(
    X_train, y_train, verbose=0
)

print("\n--- Training Evaluation ---")
print(f"Training Loss: {train_loss:.4f}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# 7. Evaluate test accuracy and loss
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print("\n--- Test Evaluation ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 8. Predict classes for 10 sample test images
sample_images = X_test[:10]
actual_labels = y_test[:10]

probabilities = model.predict(sample_images, verbose=0)
predicted_labels = np.argmax(probabilities, axis=1)

print("\n--- Sample Predictions ---")
for i in range(10):
    print(
        f"Image {i + 1}: "
        f"Actual = {class_names[actual_labels[i]]}, "
        f"Predicted = {class_names[predicted_labels[i]]}"
    )

# 9. Visualize predictions
plt.figure(figsize=(14, 7))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap="gray")

    actual = class_names[actual_labels[i]]
    predicted = class_names[predicted_labels[i]]

    color = "green" if actual_labels[i] == predicted_labels[i] else "red"

    plt.title(
        f"Pred: {predicted}\nActual: {actual}",
        color=color,
        fontsize=9
    )
    plt.axis("off")

plt.tight_layout()
plt.show()
