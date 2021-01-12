import numpy as np
from numba import jit

from tools.Math import Math


class Layer(Math):
    size: int
    input_size: int
    biases: np.array
    weights: np.array

    def __init__(self, size: int, input_size: int, activation, derivative):
        self.size = size
        self.input_size = input_size
        self.biases = np.random.randn(size)
        self.weights = np.random.randn(size, input_size)
        self.pre_activation = self._weighted_sum
        self.activation = activation
        self.activation_prime = derivative
        self.l_m = np.zeros((size, input_size))
        self.l_v = np.zeros((size, input_size))

    """
    Public Methods
    """

    def forward(self, data: np.ndarray):
        aggregation = self.pre_activation(data, self.weights, self.biases)
        activation = self.activation(aggregation)
        return activation

    @staticmethod
    @jit(nopython=True)
    def update_weights(weights, gradient: np.ndarray, learning_rate: float):
        """
        Gradient Descent Update
        """
        weights -= learning_rate * (0.01 * gradient)

    @staticmethod
    @jit(nopython=True)
    def update_biases(biases, gradient: np.ndarray, learning_rate: float):
        """
        Gradient Descent Update
        """
        biases -= learning_rate * (0.01 * gradient)
