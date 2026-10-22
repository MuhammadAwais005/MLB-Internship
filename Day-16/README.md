# Day 16: Deep Learning

## What is Deep Learning?

Deep learning is a part of machine learning that uses neural networks with
multiple layers to learn patterns from data. These layers can build on one
another to learn increasingly complex features. Deep learning is commonly used
for tasks such as image recognition, speech processing, and natural language
processing.

## Machine Learning and Deep Learning

| Machine Learning | Deep Learning |
| --- | --- |
| Includes many approaches, such as decision trees, linear models, and neural networks. | A branch of machine learning based on neural networks with multiple layers. |
| Often relies on people to select or engineer useful input features. | Can learn useful feature representations from raw data, given enough data and training. |
| Can work well with smaller datasets and less computing power, depending on the method. | Often benefits from larger datasets and more computing power. |

Deep learning is not separate from machine learning; it is one family of
machine-learning methods.

## What is a Perceptron?

A perceptron is a simple neural-network unit. It multiplies each input by a
weight, adds the results and a bias, then passes that value through an
activation function:

`output = activation(weighted_sum(inputs) + bias)`

Neural networks connect many such units in layers. A single perceptron can
model only a limited set of patterns, while networks with hidden layers can
learn more complex relationships.

## Activation Functions Explored

The practice files explored ReLU, sigmoid, and tanh. The Fashion-MNIST model
also uses softmax for its multi-class output.

| Function | Common use |
| --- | --- |
| **ReLU** | Commonly used in hidden layers of deep networks. It keeps positive values and sets negative values to zero. |
| **Sigmoid** | Maps values to a range between 0 and 1; commonly used in the output layer for binary classification. |
| **Tanh** | Maps values to a range between -1 and 1; sometimes used in hidden layers, including some recurrent networks. |
| **Softmax** | Commonly used in the output layer for multi-class classification to produce class probabilities. |

In `practice2_neural_network.py`, the hidden layer uses ReLU and the single
output uses sigmoid for a binary-classification example. In
`practice3_activation_comparison.py`, ReLU, sigmoid, and tanh are compared as
hidden-layer activations. The Fashion-MNIST model uses ReLU in its hidden layer
and softmax in its 10-class output layer.

## Fashion-MNIST Model and Accuracy

The Fashion-MNIST model in `fashion_mnist_ann.py` classifies grayscale images
of clothing into 10 categories. Its architecture is:

1. A `28 × 28` image input, flattened into 784 values.
2. A dense hidden layer with 128 units and ReLU activation.
3. A 10-unit output layer with softmax activation.

The model is trained with Adam, sparse categorical cross-entropy, a batch size
of 32, and 10 epochs. It uses 20% of the training data for validation.

| Evaluation set | Accuracy |
| --- | ---: |
| Training data (60,000 images) | **91.28%** |
| Test data (10,000 images) | **88.04%** |

These figures were measured by evaluating the saved `fashion_mnist_ann.keras`
model on the complete Fashion-MNIST training and test sets. The training
accuracy is therefore a full-training-set evaluation; it is not the final
epoch's training metric from `model.fit`, which trains on the 80% training
split when `validation_split=0.2`.
