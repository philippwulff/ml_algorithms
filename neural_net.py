import sys
import numpy as np
import matplotlib
from ml_algorithms.data import spiral_data


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.matmul(inputs, self.weights) + self.biases


class ActivationReLU:
    """
    The activation function is used to further modify a neuron's output, besides the weights and bias.
    It uses the output from the matrix multiplication as input and its output will be the output of the neuron.
    If no activation function were to be specified, it would in fact be linear. For the entire neural network, this
    means, that it can only fit a linear solution (a straight line) onto a data set. Using any non-linear function as an
    activation function solves this (in a neural net with a least two hidden layers).
    It also holds the name "activation" function, because it can define the "area of effect" for certain neurons
    (see ReLUs). We say, a neuron is activated, when the output of its activation function is greater than zero. Only
    activated neurons can have an impact on the output of the entire neural net.

    -> In order to fit non-linear problems with neural networks, we need two or more hidden layers and non-linear
       activation functions.

    This is a good graphical explanation:
    https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5&ab_channel=sentdex
    """
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    """
    The softmax activation function exponentiates and then normalizes the output values from a layer of neurons.
    The softmax layer is used as the last layer in the neural net.
    Thus: input -> exponentiate -> normalize -> output.
    This is done because, we want to interpret the values of the neurons as a probability of their correctness. When
    using ReLUs, negative values would be set to zero and thus their information would be lost. With exponentiation of
    the negative values, these neurons' values become very small but never zero.
    Also to avoid an overflow (too large numbers), before exponentiation, the largest value in a batch is substracted
    from the smallest. This does not impact the output after normalization, but prevents overflow.
    However, since the softmax layer cannot output negative values, it cannot replace ReLUs used in hidden layers.
    """
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)
print("Data shape: ", X.shape)

layer1 = LayerDense(2, 3)
activation1 = ActivationReLU()

layer2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])




