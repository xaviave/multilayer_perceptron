import numpy as np

from numba import jit


class Math:
    """
    PREPARATION OF DATASET
    """

    @staticmethod
    @jit(nopython=True)
    def standardize(X):
        return (X - np.mean(X)) / np.std(X)

    @staticmethod
    @jit(nopython=True)
    def normalize(X):
        return (X - np.min(X)) / (np.max(X) - np.min(X))

    @staticmethod
    @jit(nopython=True)
    def _weighted_sum(X, W, B):
        return np.dot(W, X) + B

    """
    ACTIVATION
    """

    @staticmethod
    @jit(nopython=True)
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    @jit(nopython=True)
    def tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    @jit(nopython=True)
    def relu(z):
        return np.array([max(0, zi) for zi in z])

    @staticmethod
    @jit(nopython=True)
    def leaky_relu(z):
        return np.array([zi if zi > 0 else 0.2 for zi in z])

    @jit(nopython=True)
    def prelu(self, z):
        return np.array([zi if zi > 0 else self.learning_rate * zi for zi in z])

    """
    DERIVATIVE
    """

    @staticmethod
    @jit(forceobj=True)
    def d_sigmoid(z):
        return Math.sigmoid(z) * (1 - Math.sigmoid(z))

    @staticmethod
    @jit(forceobj=True)
    def d_tanh(z):
        return 1 - np.power(Math.tanh(z), 2)

    @staticmethod
    @jit(nopython=True)
    def d_relu(z):
        return np.array([1 if zi > 0 else 0 for zi in z])

    @staticmethod
    @jit(nopython=True)
    def d_leaky_relu(z):
        return np.array([1 if zi > 0 else 0.2 for zi in z])

    @jit(nopython=True)
    def d_prelu(self, z):
        return np.array([1 if zi > 0 else self.learning_rate for zi in z])

    """
    ERROR
    """

    @staticmethod
    @jit(nopython=True)
    def mean_squared(Y, Z):
        return 1.0 / Z.shape[0] * np.sum(np.power(Y - Z, 2))

    @staticmethod
    @jit(nopython=True)
    def cross_entropy(Y, Z):
        epsilon = 1e-5
        return -(1.0 / Z.shape[0]) * np.sum(
            Y * np.log(Z + epsilon) + (1 - Y) * np.log(1 - Z + epsilon)
        )

    """
    OUTPUT
    """

    @staticmethod
    @jit(nopython=True)
    def soft_max(Z):
        return np.exp(Z) / np.sum(np.exp(Z))
