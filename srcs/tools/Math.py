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
        a = np.dot(W, X)
        return a + B

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
    def leaky_relu(z, alpha):
        z[z < alpha] = alpha
        return z

    def relu(self, z):
        return self.leaky_relu(z, 0)

    """
    DERIVATIVE
    """

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def d_tanh(self, z):
        return 1 - np.power(self.tanh(z), 2)

    @staticmethod
    def d_leakyrelu(z, alpha):
        return np.where(z > 0, 1, alpha)
        # return 1 if z > 0 else alpha

    def d_relu(self, z):
        return self.d_leakyrelu(z, 0)

    """
    ERROR
    """

    @staticmethod
    def mean_squared(Y, Z):
        return 1.0 / Z.shape[0] * np.sum(np.power(Y - Z, 2))

    @staticmethod
    def cross_entropy(Y, Z):
        a = -(1 / Z.shape[0])
        b = np.dot(np.log(Z), Y.T)
        c = np.dot(np.log(1 - Z), (1 - Y).T)
        return a * b + c

    """
    OUTPUT
    """

    @staticmethod
    def soft_max(Z):
        return np.exp(Z) / np.sum(np.exp(Z))
