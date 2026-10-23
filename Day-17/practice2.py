
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# 2. Normalize the images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 3. Add the channel dimension for CNN
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 4. Build a simple CNN model
model = tf.keras.Sequential([
    # Convolution Layer
    tf.keras.layers.Conv2D(
        32, kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # Max Pooling Layer
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten Layer
    tf.keras.layers.Flatten(),

    # Dense Hidden Layer
    tf.keras.layers.Dense(128, activation="relu"),

    # Output Layer: 10 clothing classes
    tf.keras.layers.Dense(10, activation="softmax")
])

# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Display the model architecture
model.summary()

# 7. Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# 8. Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=2
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy Percentage: {test_accuracy * 100:.2f}%")

# 9. Visualize training progress
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
model.save("fashion_cnn.keras")

print("Model saved successfully!")
