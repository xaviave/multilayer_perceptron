import numpy as np

from numba import jit


class Math:
    """
    PREPARATION OF DATASET
    """

    @staticmethod
    @jit(nopython=True)
    def standardize(X: np.ndarray):
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return X

    @staticmethod
    @jit(nopython=True)
    def normalize(X: np.ndarray):
        return (X - np.min(X)) / (np.max(X) - np.min(X))

    @staticmethod
    @jit(nopython=True)
    def _weighted_sum(X: np.ndarray, W: np.ndarray, B: np.ndarray):
        return np.dot(X, W) + B

    """
    ACTIVATION
    """

    @staticmethod
    @jit(nopython=True)
    def sigmoid(z: np.ndarray):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    @jit(nopython=True)
    def tanh(z: np.ndarray):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    @jit(nopython=True)
    def relu(z: np.ndarray):
        return np.array([max(0, zi) for zi in z])

    @staticmethod
    @jit(nopython=True)
    def leaky_relu(z: np.ndarray):
        return np.array([zi if zi > 0 else 0.2 for zi in z])

    def prelu(self, z: np.ndarray):
        return np.array([zi if zi > 0 else self.learning_rate * zi for zi in z])

    """
    DERIVATIVE
    """

    @staticmethod
    @jit(nopython=True)
    def d_sigmoid(z: np.ndarray):
        sig_z = 1.0 / (1.0 + np.exp(-z))
        return sig_z * (1 - sig_z)

    @staticmethod
    @jit(nopython=True)
    def d_tanh(z: np.ndarray):
        return 1 - np.power((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)), 2)

    @staticmethod
    @jit(nopython=True)
    def d_relu(z: np.ndarray):
        return np.array([1 if zi > 0 else 0 for zi in z])

    @staticmethod
    @jit(nopython=True)
    def d_leaky_relu(z: np.ndarray):
        return np.array([1 if zi > 0 else 0.2 for zi in z])

    def d_prelu(self, z: np.ndarray):
        return np.array([1 if zi > 0 else self.learning_rate for zi in z])

    """
    REGULARIZATION
    """

    @staticmethod
    @jit(nopython=True)
    def l1_laplacian(W: np.ndarray):
        return 0.01 * np.sum(np.abs(W))

    @staticmethod
    @jit(nopython=True)
    def l2_gaussian(W: np.ndarray):
        return 0.01 * np.sum(np.power(W, 2))

    """
    ERROR
    """

    @staticmethod
    @jit(nopython=True)
    def mean_squared(Y: np.ndarray, Z: np.ndarray):
        return 1.0 / Z.shape[0] * np.sum(np.power(Y - Z, 2))

    @staticmethod
    @jit(nopython=True)
    def cross_entropy(Y: np.ndarray, Z: np.ndarray):
        epsilon = 1e-5
        return -(1.0 / Z.shape[0]) * np.sum(
            Y * np.log(Z + epsilon) + (1 - Y) * np.log(1 - Z + epsilon)
        )

    """
    OUTPUT
    """

    @staticmethod
    #@jit(nopython=True)
    def soft_max(z: np.ndarray):
        a = np.zeros((z.shape))
        for i, zi in enumerate(z):
            a[i] = np.exp(zi) / np.sum(np.exp(zi))
        return a

    """
    OPTIMIZATION UTILS
    """

    @staticmethod
    @jit(nopython=True)
    def get_output_delta(a: np.ndarray, target: np.ndarray):
        return a - target

    @staticmethod
    @jit(nopython=True)
    def get_deltas(activation_prime: np.ndarray, W: np.ndarray, last_delta: np.ndarray):
        return activation_prime * np.dot(W.T, last_delta)

    @staticmethod
    @jit(nopython=True)
    def get_weight_gradient(delta: np.ndarray, prev_activation: np.ndarray):
        return np.dot(delta, prev_activation)
