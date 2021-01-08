import numpy as np


class Math:
    """
    PREPARATION OF DATASET
    """

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

    """
    DERIVATIVE
    """

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

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
