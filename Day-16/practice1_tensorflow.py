
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras imported successfully!")
print("Keras version:", keras.__version__)

# Create a small tensor
tensor = tf.constant([1, 2, 3, 4, 5])
print("Tensor:", tensor)

# Perform a simple TensorFlow operation
result = tf.reduce_sum(tensor)
print("Sum:", result.numpy())
