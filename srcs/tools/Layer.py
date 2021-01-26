import numpy as np
from tools.Math import Math
from tools.Optimizer import Optimizer


class Layer(Math, Optimizer):
    size: int
    input_size: int
    biases: np.array
    weights: np.array

    def __init__(
        self,
        size: int,
        input_size: int,
        activation,
        derivative,
        optimizer,
        regularization,
    ):
        self.size = size
        self.input_size = input_size
        self.biases = np.random.randn(size)
        self.weights = np.random.randn(size, input_size)
        self.activation = activation
        self.activation_prime = derivative
        self.regularization = regularization
        self.optimizer = optimizer
        self.optimizer_value = {
            "w": {"momentum": 0, "rms_prop": 0},
            "b": {"momentum": 0, "rms_prop": 0},
        }

    """
    Public Methods
    """

    def update_weights(
        self, weights: np.ndarray, gradient: np.ndarray, learning_rate: float
    ):
        """
        Gradient Descent Update
        """
        weights -= self.optimizer(
            self.optimizer_value["w"],
            gradient,
            learning_rate,
            self.regularization(weights),
        )

    def update_biases(
        self, biases: np.ndarray, gradient: np.ndarray, learning_rate: float
    ):
        """
        Gradient Descent Update
        """
        biases -= self.optimizer(
            self.optimizer_value["b"],
            gradient,
            learning_rate,
            self.regularization(biases),
        )
