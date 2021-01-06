import logging
import datetime

import numpy as np
import matplotlib.pyplot as plt

from tools.Layer import Layer
from tools.DataPreprocessing import DataPreprocessing

from sklearn.utils import shuffle


class Network(DataPreprocessing):
    input_dim: int
    deltas: list
    layers: list
    loss: list = []
    val_loss: list = []
    times: list = []
    predicted: list = []
    activations: list = []
    weighted_sums: list = []
    default_model_file: str = "data/models/default_model"

    @staticmethod
    def _to_one_hot(y: int, k: int) -> np.array:
        """
        Convertit un entier en vecteur "one-hot".
        to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0)
        """
        one_hot = np.zeros(k)
        one_hot[y] = 1
        return one_hot

    def _add_layer(self, size: int):
        """
        Create and Append a new layer with size (size: int) to the Neural Network
        """
        input_layer_dim = (
            self.layers[-1].size if len(self.layers) > 0 else self.input_dim
        )
        self.layers.append(Layer(size, input_layer_dim))

    def _init_layers(self, input_dim, layers_size):
        self.input_dim = input_dim
        self.layers_size = layers_size
        for s in layers_size:
            self._add_layer(s)
        self.layers[-1].activation = self.soft_max

    def __init__(self, input_dim: int = None, layers_size: list = None):
        self.layers = []
        if input_dim and layers_size:
            self._init_layers(input_dim, layers_size)
        super().__init__()
        self.wbdc_preprocess()

    def _visualize(self, epochs):
        plt.subplot(1, 2, 1)
        plt.plot(range(epochs), self.loss, c="r")
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), self.val_loss, c="b")
        plt.show()

    def get_output_delta(self, a, target):
        return a - target

    def feedforward(self, x):
        self.activations = [x]
        self.weighted_sums = []
        for layer in self.layers:
            self.weighted_sums.append(
                self.pre_activation(self.activations[-1], layer.weights, layer.biases)
            )
            self.activations.append(layer.activation(self.weighted_sums[-1]))
        self.predicted.append(self.activations[-1])

    def calcul_backpropagation(self, y):
        delta = self.get_output_delta(self.activations[-1], self._to_one_hot(int(y), 2))
        deltas = [delta]

        nb_layers = len(self.layers) - 2
        for i in range(nb_layers, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            activation_prime = layer.activation_prime(self.weighted_sums[i])
            deltas.append(activation_prime * np.dot(next_layer.weights.T, deltas[-1]))
        self.deltas = list(reversed(deltas))

    def apply_backpropagation(self):
        bias_gradient = []
        weight_gradient = []
        for i in range(len(self.layers)):
            prev_activation = self.activations[i]
            weight_gradient.append(np.outer(self.deltas[i], prev_activation))
            bias_gradient.append(self.deltas[i])
        return weight_gradient, bias_gradient

    def train_batch(self, X: np.ndarray, Y: np.ndarray, learning_rate: float):
        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]
        for (x, y) in zip(X, Y):
            self.feedforward(x)
            self.calcul_backpropagation(y)
            new_weight_gradient, new_bias_gradient = self.apply_backpropagation()
            for i in range(len(self.layers)):
                weight_gradient[i] += new_weight_gradient[i]
                bias_gradient[i] += new_bias_gradient[i]

        for layer, wg, bg in zip(self.layers, weight_gradient, bias_gradient):
            layer.update_weights(wg / Y.size, learning_rate)
            layer.update_biases(bg / Y.size, learning_rate)

    def train(
        self,
        epochs: int = 30,
        learning_rate: float = 0.3,
        batch_size: int = 10,
    ):
        """
        Create (batch_size) number of random batch data
        """
        n = self.Y.size
        for e in range(epochs):
            self.predicted = []
            start = datetime.datetime.now()
            X, Y = shuffle(self.X, self.Y)
            cross_y = np.array([self._to_one_hot(y, 2) for y in Y])
            for batch_start in range(0, n, batch_size):
                X_batch, Y_batch = (
                    X[batch_start : batch_start + batch_size],
                    Y[batch_start : batch_start + batch_size],
                )
                self.train_batch(X_batch, Y_batch, learning_rate)
            self.loss.append(self.cross_entropy(cross_y, np.array(self.predicted)))
            self.times.append(datetime.datetime.now() - start)
            predicted = np.array([self.predict(x) for x in X])
            self.val_loss.append(
                np.where(predicted == Y)[0].shape[0] / predicted.shape[0]
            )
            logging.info(
                f"epoch {e + 1}/{epochs} - loss: {self.loss[-1]:.4f} - val_loss {self.val_loss[-1]:.4f} - time: {self.times[-1]}"
            )
        self._visualize(epochs)

    """
    Predict
    """

    def predict_feedforward(self, input_data):
        activation = input_data
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def predict(self, input_data):
        return np.argmax(self.predict_feedforward(input_data))

    def evaluate(self):
        results = [
            1 if self.predict(x) == y else 0 for (x, y) in zip(self.X_test, self.Y_test)
        ]
        accuracy = sum(results) / len(results)
        return accuracy

    """
    File Handling Method
    """

    def save_model(self, model_file: str = default_model_file):
        model = [(self.input_dim, self.layers_size)]
        model.extend([(l.weights.tolist(), l.biases.tolist()) for l in self.layers])
        self._save_npy(model_file, model)

    def load_model(self, model_file: str = default_model_file):
        raw_model = self._load_npy(f"{model_file}.npy")
        input_dim = raw_model[0][0]
        layers_size = [l for l in raw_model[0][1]]
        self._init_layers(input_dim, layers_size)
        for i in range(len(raw_model[0][1])):
            self.layers[-1].weights = np.array(raw_model[i + 1][0])
            self.layers[-1].biases = np.array(raw_model[i + 1][1])
