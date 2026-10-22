
import tensorflow as tf

# 1. Create the neural network
model = tf.keras.Sequential([
    # Input Layer: accepts 4 features
    tf.keras.layers.Input(shape=(4,), name="input_layer"),

    # Hidden Layer: 8 neurons with ReLU activation
    tf.keras.layers.Dense(8, activation="relu", name="hidden_layer"),

    # Output Layer: 1 neuron for a binary classification score
    tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")
])

# 2. Print the model summary
model.summary()
