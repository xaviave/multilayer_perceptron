import numpy as np


class Math:
    """"""

    """
    BASIC CALCUL
    """

    @staticmethod
    def count(collection):
        count = 0
        for _ in collection:
            count += 1
        return count

    @staticmethod
    def newton_sqrt(x):
        z = 1.0
        for _ in range(0, 10):
            z -= (z * z - x) / (2 * z)
        return z

    @staticmethod
    def mean(collection, count):
        return sum([e for e in collection]) / len(count)

    def std(self, collection, count):
        mean = self.mean(collection, count)
        v = sum([np.power(e - mean, 2) for e in collection])
        return self.newton_sqrt(v / count)

    """
    PREPARATION OF DATASET
    """

    @staticmethod
    def standardize(X, mean, std):
        return (X - mean) / std

    @staticmethod
    def pre_activation(X, W, B):
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
    def relu(z):
        return max(0, z)

    @staticmethod
    def leaky_relu(z, alpha):
        return max(alpha * z, z)

    """
    DERIVATIVE
    """

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def d_tanh(self, z):
        return 1 - np.power(self.tanh(z), 2)

    @staticmethod
    def d_relu(z):
        return 1 if z > 0 else 0

    @staticmethod
    def d_leakyrelu(z, alpha):
        return 1 if z > 0 else alpha

    """
    ERROR
    """

    @staticmethod
    def mean_squared(Y, Z):
        return 0.5 * np.power(Y - Z, 2)

    @staticmethod
    def cross_entropy(Y, Z):
        return (
            -1
            / Z.shape([0])
            * np.sum(np.dot(Y, np.log(Z)) + np.dot(1 - Y, np.log(1 - Z)))
        )

    """
    OUTPUT
    """

    @staticmethod
    def soft_max(Z):
        return np.exp(Z) / np.sum(np.exp(Z))
