import copy
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tabulate import tabulate

from tools.Layer import Layer
from tools.DataPreprocessing import DataPreprocessing


class Network(DataPreprocessing):
    input_dim: int
    deltas: list
    layers: list
    loss: list = []
    times: list = []

    val_loss: list = []
    predicted: list = []
    activations: list = []
    layers_size: list = []
    weighted_sums: list = []
    best_acc: list = [0, 0]
    best_loss: list = [0, 0]
    default_model_file: str = "data/models/default_model"

    """
    Override Methods
    """

    def _activation_func_arg(self, parser):
        activation_group = parser.add_mutually_exclusive_group(required=False)
        activation_group.add_argument(
            "-sac",
            "--sigmoid",
            action="store_const",
            const={"activation": self.sigmoid, "derivative": self.d_sigmoid},
            help="Use sigmoid as activation function (default value)",
            dest="type_activation",
        )
        activation_group.add_argument(
            "-tac",
            "--tanh",
            action="store_const",
            const={"activation": self.tanh, "derivative": self.d_tanh},
            help="Use tanh as activation function",
            dest="type_activation",
        )
        activation_group.add_argument(
            "-rac",
            "--relu",
            action="store_const",
            const={"activation": self.relu, "derivative": self.d_relu},
            help="Use ReLU as activation function",
            dest="type_activation",
        )
        activation_group.add_argument(
            "-lac",
            "--leaky_relu",
            action="store_const",
            const={"activation": self.leaky_relu, "derivative": self.d_leaky_relu},
            help="Use leaky ReLU as activation function",
            dest="type_activation",
        )
        activation_group.add_argument(
            "-pac",
            "--parametric_relu",
            action="store_const",
            const={"activation": self.prelu, "derivative": self.d_prelu},
            help="Use parametric ReLU as activation function",
            dest="type_activation",
        )

    def _add_parser_args(self, parser):
        super()._add_parser_args(parser)
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_const",
            const=1,
            help=f"Add more evaluation metrics",
            dest="verbose",
        )
        self._activation_func_arg(parser)

    """
    Private Methods
    """

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
        self.layers.append(
            Layer(
                size,
                input_layer_dim,
                activation=self.activation_func,
                derivative=self.derivative,
            )
        )

    def _init_layers(self, input_dim, layers_size):
        self.input_dim = input_dim
        self.layers_size = layers_size
        for s in layers_size:
            self._add_layer(s)
        self.layers[-1].activation = self.soft_max

    def _shuffle(self, X, Y):
        c = np.c_[self.X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
        np.random.shuffle(c)
        return (
            c[:, : X.size // len(X)].reshape(X.shape),
            c[:, X.size // len(X) :].reshape(Y.shape),
        )

    def _visualize(self, epochs):
        fig = plt.figure(figsize=(10, 7))
        ep = range(self.epochs)
        val_loss = [v * 100 for v in self.val_loss]

        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.xlabel("epochs")
        (line,) = plt.plot([], [], c="r")
        plt.ylim(min(self.loss) - 0.2, max(self.loss) + 0.2)
        plt.xlim(-10, epochs + 10)

        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("percents")
        (line2,) = plt.plot([], [], c="b")
        plt.ylim(min(val_loss) - 5, max(val_loss) + 5)
        plt.xlim(-10, epochs + 10)

        def animate(i):
            line.set_data(ep[:i], self.loss[:i])
            line2.set_data(ep[:i], val_loss[:i])
            return (
                line,
                line2,
            )

        _ = animation.FuncAnimation(
            fig, animate, frames=epochs, blit=True, interval=1, repeat=False
        )
        plt.show()

    @staticmethod
    def _f1_score(TP, FP, FN):
        precision = TP / (TP + FP)
        rappel = TP / (TP + FN)
        return 2 / ((1 / precision) + (1 / rappel))

    def _additional_metrics(self, predicted, Y):
        TP = np.where((Y == predicted) & (Y == 1))[0].shape[0]
        FP = np.where((Y != predicted) & (predicted == 1))[0].shape[0]
        TN = np.where((Y == predicted) & (Y == 0))[0].shape[0]
        FN = np.where((Y != predicted) & (predicted == 0))[0].shape[0]
        print(
            f"F1 score = {self._f1_score(TP, FP, FN) if TP + FP != 0 and TP + FN != 0 else 'nan'}",
            f"Mean squared Error = {self.mean_squared(Y, predicted)}\n",
            "Confusion Matrix\n",
            tabulate(
                [["False", TN, FP], ["True", FN, TP]],
                headers=[f"sample size={Y.shape[0]}", "False", "True"],
            ),
            f"\n{'-'*70}",
        )

    @staticmethod
    def _get_output_delta(a, target):
        return a - target

    def _feedforward(self, x):
        self.activations = [x]
        self.weighted_sums = []
        for layer in self.layers:
            self.weighted_sums.append(
                self._weighted_sum(self.activations[-1], layer.weights, layer.biases)
            )
            self.activations.append(layer.activation(self.weighted_sums[-1]))
        self.predicted.append(self.activations[-1])

    def _calcul_backpropagation(self, y):
        delta = self._get_output_delta(
            self.activations[-1], self._to_one_hot(int(y), 2)
        )
        deltas = [delta]

        nb_layers = len(self.layers) - 2
        for i in range(nb_layers, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            activation_prime = layer.activation_prime(self.weighted_sums[i])
            deltas.append(activation_prime * np.dot(next_layer.weights.T, deltas[-1]))
        self.deltas = list(reversed(deltas))

    def _apply_backpropagation(self):
        bias_gradient = []
        weight_gradient = []
        for i in range(len(self.layers)):
            prev_activation = self.activations[i]
            weight_gradient.append(np.outer(self.deltas[i], prev_activation))
            bias_gradient.append(self.deltas[i])
        return weight_gradient, bias_gradient

    def _train_batch(self, X: np.ndarray, Y: np.ndarray, learning_rate: float):
        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]
        for (x, y) in zip(X, Y):
            self._feedforward(x)
            self._calcul_backpropagation(y)
            new_weight_gradient, new_bias_gradient = self._apply_backpropagation()
            for i in range(len(self.layers)):
                weight_gradient[i] += new_weight_gradient[i]
                bias_gradient[i] += new_bias_gradient[i]

        for layer, wg, bg in zip(self.layers, weight_gradient, bias_gradient):
            layer.update_weights(wg / Y.size, learning_rate)
            layer.update_biases(bg / Y.size, learning_rate)

    def _evaluate(self, start, X, Y, e: int = None, epochs: int = None):
        cross_y = np.array([self._to_one_hot(int(y), 2) for y in Y])
        self.loss.append(self.cross_entropy(cross_y, np.array(self.predicted)))
        self.times.append(datetime.datetime.now() - start)
        predicted = np.array([self._predict(x) for x in X])
        if self.verbose:
            self._additional_metrics(predicted, Y)
        well = np.where(predicted == Y)[0]
        self.val_loss.append(well.shape[0] / Y.shape[0])
        print(
            f"epoch {e + 1 if e is not None else None}/{epochs} - loss: {self.loss[-1]:.4f} - val_loss {self.val_loss[-1]:.4f} - time: {self.times[-1]}"
        )

    """
    Predict
    """

    def _predict_feedforward(self, input_data):
        activation = input_data
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def _predict(self, input_data):
        return np.argmax(self._predict_feedforward(input_data))

    def __init__(
        self,
        input_dim: int = None,
        layers_size: list = None,
        epochs: int = 100,
        learning_rate: float = 0.5,
        name: str = "main"
    ):
        print(
            "\033[92m",
            f"{name} init_network",
            input_dim,
            layers_size,
            epochs,
            learning_rate,
            "\033[0m",
        )
        self.name = name
        self.layers = []
        super().__init__()
        self.wbdc_preprocess()
        self.verbose = self.get_args("verbose", default_value=0)
        self.activation_func, self.derivative = self.get_args(
            "type_activation",
            default_value={"activation": self.sigmoid, "derivative": self.d_sigmoid},
        ).values()
        if input_dim is not None and layers_size is not None:
            self._init_layers(input_dim, layers_size)
        self.epochs = epochs
        self.learning_rate = learning_rate

    """
    Public Methods
    """

    def train(self, batch_size: int = 10):
        n = self.Y.size
        watch_perf = int(self.epochs - (self.epochs / 10))
        for e in range(self.epochs):
            self.predicted = []
            start = datetime.datetime.now()
            X, Y = self._shuffle(self.X, self.Y)
            for batch_start in range(0, n, batch_size):
                X_batch, Y_batch = (
                    X[batch_start : batch_start + batch_size],
                    Y[batch_start : batch_start + batch_size],
                )
                self._train_batch(X_batch, Y_batch, self.learning_rate)
            self._evaluate(start, X, Y, e=e, epochs=self.epochs)
            if e > watch_perf and self.best_acc[0] < self.val_loss[-1]:
                self.best_acc = [self.val_loss[-1], copy.deepcopy(self.layers)]
            if e > watch_perf and self.best_loss[0] < self.loss[-1]:
                self.best_loss = [self.loss[-1], copy.deepcopy(self.layers)]
        self.layers = self.best_loss[1] if self.best_loss[1] != 0 else self.layers
        # print(f"{self.name} finish")
        #self._visualize(self.epochs)

    def evaluate(self):
        start = datetime.datetime.now()
        X, Y = self._shuffle(self.X_test, self.Y_test)
        self.predicted = [self._predict_feedforward(x) for x in X]
        self._evaluate(start, X, Y)

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
        for i in range(len(layers_size)):
            self.layers[i].weights = np.array(raw_model[i + 1][0])
            self.layers[i].biases = np.array(raw_model[i + 1][1])
