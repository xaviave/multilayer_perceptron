import numpy as np


class Math:
    """
    PREPARATION OF DATASET
    """

    @staticmethod
    def standardize(X):
        return (X - np.mean(X)) / np.std(X)

    @staticmethod
    def normalize(X):
        return (X - np.min(X)) / (np.max(X) - np.min(X))

    @staticmethod
    def _weighted_sum(X, W, B):
        return np.dot(W, X) + B

    """
    ACTIVATION
    """

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def leaky_relu(z):
        return np.array([max(0.2, zi) for zi in z])

    def relu(self, z):
        return np.array([max(0, zi) for zi in z])

    """
    DERIVATIVE
    """

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def d_tanh(self, z):
        return 1 - np.power(self.tanh(z), 2)

    @staticmethod
    def d_leaky_relu(z):
        return np.array([1 if zi > 0 else 0.2 for zi in z])

    def d_relu(self, z):
        return np.array([1 if zi > 0 else 0 for zi in z])

    """
    ERROR
    """

    @staticmethod
    def mean_squared(Y, Z):
        return 1.0 / Z.shape[0] * np.sum(np.power(Y - Z, 2))

    @staticmethod
    def cross_entropy(Y, Z):
        return -(1.0 / Z.shape[0]) * np.sum(Y * np.log(Z) + (1 - Y) * np.log(1 - Z))

    """
    OUTPUT
    """

    @staticmethod
    def soft_max(Z):
        return np.exp(Z) / np.sum(np.exp(Z))
