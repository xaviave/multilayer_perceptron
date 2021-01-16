import numpy as np

from numba import jit


class Math:
    """
    PREPARATION OF DATASET
    """

    @staticmethod
    @jit(nopython=True)
    def standardize(X):
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return X

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
    @jit(nopython=True)
    def d_sigmoid(z):
        sig_z = 1.0 / (1.0 + np.exp(-z))
        return sig_z * (1 - sig_z)

    @staticmethod
    @jit(nopython=True)
    def d_tanh(z):
        return 1 - np.power((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)), 2)

    @staticmethod
    @jit(nopython=True)
    def d_relu(z):
        return np.array([1 if zi > 0 else 0 for zi in z])

    @staticmethod
    @jit(nopython=True)
    def d_leaky_relu(z):
        return np.array([1 if zi > 0 else 0.2 for zi in z])

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

    """
    OPTIMIZATION UTILS
    """

    @staticmethod
    @jit(nopython=True)
    def get_output_delta(a, target):
        return a - target

    @staticmethod
    @jit(nopython=True)
    def get_deltas(activation_prime, W, last_delta):
        return activation_prime * np.dot(W.T, last_delta)

    @staticmethod
    @jit(nopython=True)
    def get_weight_gradient(delta, prev_activation):
        return np.outer(delta, prev_activation)
