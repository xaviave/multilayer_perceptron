import datetime

import numpy as np
import pandas as pd

from Math import Math

from sklearn.utils import shuffle


def load_mnist_data(path: str) -> tuple:
    train_data = pd.read_csv(path, header=None)
    np.random.shuffle(train_data.values)
    labels = train_data.pop(0)
    imgs = train_data.to_numpy() / 255
    # imgs = imgs.reshape(imgs.shape[0], 28, 28)
    return imgs, labels


class Layer(Math):
    size: int
    input_size: int
    biases: np.array
    weights: np.array

    def __init__(self, size: int, input_size: int):
        self.size = size
        self.input_size = input_size
        self.biases = np.random.randn(size)
        self.weights = np.random.randn(size, input_size)
        self.activation = self.sigmoid
        self.activation_prime = self.d_sigmoid

    def forward(self, data: np.ndarray):
        aggregation = self.pre_activation(data, self.weights, self.biases)
        activation = self.activation(aggregation)
        return activation

    def update_weights(self, gradient: np.ndarray, learning_rate: float):
        """
        Gradient Descent Update
        """
        self.weights -= learning_rate * gradient

    def update_biases(self, gradient: np.ndarray, learning_rate: float):
        """
        Gradient Descent Update
        """
        self.biases -= learning_rate * gradient


class Network(Math):
    deltas: list
    layers: list
    input_dim: int
    activations: list = []
    weighted_sums: list = []

    @staticmethod
    def to_one_hot(y: int, k: int) -> np.array:
        """
        Convertit un entier en vecteur "one-hot".
        to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0)
        """
        one_hot = np.zeros(k)
        one_hot[y] = 1
        return one_hot

    def add_layer(self, size: int):
        """
        Create and Append a new layer with size (size: int) to the Neural Network
        """
        input_layer_dim = (
            self.layers[-1].size if len(self.layers) > 0 else self.input_dim
        )
        self.layers.append(Layer(size, input_layer_dim))

    def __init__(self, input_dim: int):
        self.layers = []
        self.input_dim = input_dim

    # Cf http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function
    def get_output_delta(self, a, target):
        """
        Output Error Delta
        """
        return a - target

    def feedforward(self, x):
        self.activations = [x]
        self.weighted_sums = []
        for layer in self.layers:
            self.weighted_sums.append(
                self.pre_activation(self.activations[-1], layer.weights, layer.biases)
            )
            self.activations.append(layer.activation(self.weighted_sums[-1]))

    def calcul_backpropagation(self, y):
        delta = self.get_output_delta(self.activations[-1], self.to_one_hot(int(y), 10))
        deltas = [delta]

        nb_layers = len(self.layers) - 2
        for i in range(nb_layers, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i - 1]
            activation_prime = layer.activation_prime(self.weighted_sums[i])
            deltas.append(activation_prime * np.dot(next_layer.weights.T, delta))
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
        """
        A. Init the weight and bias gradient's matrices
        B. [Weighted Sum]   1. Launch Backpropagation adding up every resulting Weight and Bias Matrices
                            2. Mean of every Weight and Bias Matrices
        C. Update Weights and Bias Matrices
        """
        # A
        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]
        # B - 1
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
        X: np.ndarray,
        Y: np.ndarray,
        steps: int = 30,
        learning_rate: float = 0.3,
        batch_size: int = 10,
    ):
        """
        Create (batch_size) number of random batch data
        """
        n = Y.size
        for _ in range(steps):
            X, Y = shuffle(X, Y)
            for batch_start in range(0, n, batch_size):
                X_batch, Y_batch = (
                    X[batch_start : batch_start + batch_size],
                    Y[batch_start : batch_start + batch_size],
                )
                self.train_batch(X_batch, Y_batch, learning_rate)

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

    def evaluate(self, X: np.ndarray, Y: np.ndarray):
        results = [1 if self.predict(x) == y else 0 for (x, y) in zip(X, Y)]
        accuracy = sum(results) / len(results)
        return accuracy


if __name__ == "__main__":
    start = datetime.datetime.now()
    data_path = "data/datasets/mnist/"
    X_train, Y_train = load_mnist_data(data_path + "mnist_train.csv")
    X_test, Y_test = load_mnist_data(data_path + "mnist_test.csv")
    print(f"data loaded: {datetime.datetime.now() - start}")
    net = Network(input_dim=784)
    net.add_layer(200)
    net.add_layer(10)
    tps = []
    perfs = []
    for i in range(1):
        start = datetime.datetime.now()
        net.train(X_train, Y_train, steps=1, learning_rate=1)
        accuracy = net.evaluate(X_test, Y_test)
        perfs.append(accuracy)
        tps.append(datetime.datetime.now() - start)
        print(f"Nouvelle performance : {perfs[-1] * 100}% | time: {tps[-1]}")
    print(f"Mean perf : {np.mean(perfs) * 100}% | time: {np.mean(tps)}")
