
import tensorflow as tf



def build_model(activation_name):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,), name="input_layer"),
        tf.keras.layers.Dense(
            8,
            activation=activation_name,
            name="hidden_layer"
        ),
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="output_layer"
        )
    ])
    return model


activations = ["relu", "sigmoid", "tanh"]

for activation in activations:
    print("\n" + "=" * 50)
    print(f"Hidden activation: {activation.upper()}")

    model = build_model(activation)
    model.summary()



#test to check if the activation functions are working correctly

x = tf.constant([-2.0, 0.0, 2.0])

print("ReLU:", tf.nn.relu(x).numpy())
print("Sigmoid:", tf.nn.sigmoid(x).numpy())
print("Tanh:", tf.nn.tanh(x).numpy())
