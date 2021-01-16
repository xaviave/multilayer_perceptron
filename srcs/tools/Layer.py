import numpy as np
from numba import jit
from tools.Math import Math


class Layer(Math):
    size: int
    input_size: int
    biases: np.array
    weights: np.array

    def __init__(self, size: int, input_size: int, activation, derivative):
        self.mom_w = 0
        self.mom_b = 0
        self.rms_w = 0
        self.rms_b = 0
        self.size = size
        self.input_size = input_size
        self.biases = np.random.randn(size)
        self.weights = np.random.randn(size, input_size)
        self.pre_activation = self._weighted_sum
        self.activation = activation
        self.activation_prime = derivative

    """
    Public Methods
    """

    def forward(self, data: np.ndarray):
        aggregation = self.pre_activation(data, self.weights, self.biases)
        activation = self.activation(aggregation)
        return activation

    # @staticmethod
    # @jit(nopython=True)
    def momentum(self, mom, gradient: np.ndarray):
        """
        Momentum Optimization
        """
        previous_momentum = mom
        mom = (0.01 * gradient) * 0.1 + previous_momentum * 0.9
        return mom

    def adam(self, mom, rms, gradient, learning_rate):
        """
        Adam Optimization
        """
        gamma = 1e-5
        previous_momentum = mom
        previous_rms = rms
        mom = (0.01 * gradient) * 0.1 + previous_momentum * 0.9
        rms = np.power(gradient, 2) * 0.1 + previous_rms * 0.9
        return (mom * learning_rate) / (np.sqrt(rms) + gamma)

    def rms_prop(self, rms, gradient: np.ndarray, learning_rate):
        """
        RMSprop Optimization
        """
        gamma = 1e-5
        previous_rms = rms
        rms = np.power(gradient, 2) * 0.1 + previous_rms * 0.9
        return (gradient * 0.01 * learning_rate) / (np.sqrt(rms) + gamma)

    @staticmethod
    @jit(nopython=True)
    def sgd(gradient: np.ndarray, learning_rate: float):
        """
        Vanilla Stochastic Gradient Descent
        """
        return learning_rate * (0.01 * gradient)

    # @staticmethod
    # @jit(nopython=True)
    def update_weights(self, weights, gradient: np.ndarray, learning_rate: float):
        """
        Gradient Descent Update
        """
        # weights -= Layer.sgd(gradient, learning_rate)
        weights -= self.momentum(self.mom_w, gradient)
        # weights -= self.rms_prop(self.rms_w, gradient, learning_rate)
        # weights -= self.adam(self.mom_w, self.rms_w, gradient, learning_rate)

    # @staticmethod
    # @jit(nopython=True)
    def update_biases(self, biases, gradient: np.ndarray, learning_rate: float):
        """
        Gradient Descent Update
        """
        # biases -= Layer.sgd(gradient, learning_rate)
        biases -= self.momentum(self.mom_b, gradient)
        # biases -= self.rms_prop(self.rms_b, gradient, learning_rate)
        # biases -= self.adam(self.mom_b, self.rms_b, gradient, learning_rate)
