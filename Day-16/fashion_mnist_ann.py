
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


fashion_mnist = tf.keras.datasets.fashion_mnist


# 1. LOAD THE FASHION MNIST DATASET

print("Loading Fashion MNIST dataset...")

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# 2. EXPLORE THE DATASET

print("\n--- Dataset Information ---")
print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)
print("Original pixel range:", X_train.min(), "to", X_train.max())

# Display sample images
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.suptitle("Fashion MNIST Sample Images")
plt.tight_layout()
plt.show()


# 3. NORMALIZE THE PIXEL VALUES

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("\n--- After Normalization ---")
print("Training pixel range:", X_train.min(), "to", X_train.max())
print("Testing pixel range:", X_test.min(), "to", X_test.max())


# 4. BUILD THE ARTIFICIAL NEURAL NETWORK

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28), name="input_image"),
    tf.keras.layers.Flatten(name="flatten_layer"),
    tf.keras.layers.Dense(128, activation="relu", name="hidden_layer"),
    tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
])

print("\n--- Model Architecture ---")
model.summary()


# 5. COMPILE 

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 6. TRAIN THE MODEL

print("\n--- Starting Model Training ---")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# 7. EVALUATE ON THE TEST DATASET

print("\n--- Test Dataset Evaluation ---")

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# 8. DISPLAY TRAINING AND VALIDATION ACCURACY

print("\n--- Accuracy by Epoch ---")

for epoch in range(len(history.history["accuracy"])):
    train_acc = history.history["accuracy"][epoch]
    val_acc = history.history["val_accuracy"][epoch]

    print(
        f"Epoch {epoch + 1}: "
        f"Training Accuracy = {train_acc * 100:.2f}%, "
        f"Validation Accuracy = {val_acc * 100:.2f}%"
    )


# 9. PLOT ACCURACY AND LOSS CURVES

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 10. MAKE PREDICTIONS ON TEST IMAGES

print("\n--- Sample Predictions ---")

sample_images = X_test[:10]
actual_labels = y_test[:10]

# Predict class scores for the 10 images
predictions = model.predict(sample_images, verbose=0)

# Highest-scoring class for each image
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 6))

for i in range(10):
    predicted_name = class_names[predicted_labels[i]]
    actual_name = class_names[actual_labels[i]]

    correct = predicted_labels[i] == actual_labels[i]
    color = "green" if correct else "red"

    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i], cmap="gray")
    plt.title(
        f"Predicted: {predicted_name}\nActual: {actual_name}",
        color=color,
        fontsize=9
    )
    plt.axis("off")

    print(
        f"Image {i + 1}: "
        f"Predicted = {predicted_name}, "
        f"Actual = {actual_name}, "
        f"Correct = {correct}"
    )

plt.suptitle("Fashion MNIST: Actual vs Predicted Labels")
plt.tight_layout()
plt.show()

print("\nMini Project Completed!")

# Save the trained model
model.save("fashion_mnist_ann.keras")

print("Trained model saved successfully!")
