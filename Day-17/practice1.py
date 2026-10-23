import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Training images shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)

# 2. Define class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# 3. Visualize 10 sample images with their labels
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()

# 4. Normalize the dataset
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 5. Verify normalization
print("\nAfter normalization:")
print("Minimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
