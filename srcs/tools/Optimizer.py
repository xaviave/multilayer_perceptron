import numpy as np


class Optimizer:
    """
    Public Methods
    """

    @staticmethod
    def sgd(optimizer: dict, gradient: np.ndarray, learning_rate: float, reg: float):
        """
        Vanilla SGD Optimizer
        """
        return learning_rate * (gradient * reg)

    @staticmethod
    def momentum(
        optimizer: dict, gradient: np.ndarray, learning_rate: float, reg: float
    ):
        """
        Momentum Optimizer
        """
        previous_momentum = optimizer["momentum"]
        optimizer["momentum"] = gradient * 0.1 + previous_momentum * 0.9
        return optimizer["momentum"]

    @staticmethod
    def rms_prop(
        optimizer: dict, gradient: np.ndarray, learning_rate: float, reg: float
    ):
        """
        RMSprop Optimizer
        """
        gamma = 1e-5
        previous_rms = optimizer["rms_prop"]
        optimizer["rms_prop"] = np.power(gradient, 2) * 0.1 + previous_rms * 0.9
        return (gradient * learning_rate) / (np.sqrt(optimizer["rms_prop"]) + gamma)

    @staticmethod
    def adam(optimizer: dict, gradient: np.ndarray, learning_rate: float, reg: float):
        """
        Adam Optimizer
        """
        gamma = 1e-5
        previous_rms = optimizer["rms_prop"]
        previous_momentum = optimizer["momentum"]
        optimizer["momentum"] = gradient * 0.1 + previous_momentum * 0.9
        optimizer["rms_prop"] = np.power(gradient, 2) * 0.1 + previous_rms * 0.9
        return (optimizer["momentum"] * learning_rate) / (
            np.sqrt(optimizer["rms_prop"]) + gamma
        )
