import copy
import datetime
import logging
import warnings

import numpy as np
from numba import jit

from tools.Layer import Layer
from tools.Metrics import Metrics
from tools.Optimizer import Optimizer
from tools.DataPreprocessing import DataPreprocessing


class Network(Metrics, DataPreprocessing, Optimizer):
    input_dim: int

    deltas: list
    layers: list
    loss: list = []
    time: datetime.timedelta

    val_loss: list = []
    predicted: list = []
    activations: list = []
    layers_size: list = []
    weighted_sums: list = []
    best_loss: list = [0, 0]
    default_model_file: str = "data/models/model"

    """
    Override Methods
    """

    def _regularization_arg(self, parser):
        regularization_group = parser.add_mutually_exclusive_group(required=False)
        regularization_group.add_argument(
            "-l1",
            "--l1_laplacian",
            action="store_const",
            const=self.l1_laplacian,
            help="Use Lasso Regression as regularization function (default)",
            dest="type_regularization",
        )
        regularization_group.add_argument(
            "-l2",
            "--l2_gaussian",
            action="store_const",
            const=self.l2_gaussian,
            help="Use Ridge Regression as regularization function",
            dest="type_regularization",
        )

    def _optimizer_arg(self, parser):
        optimizer_group = parser.add_mutually_exclusive_group(required=False)
        optimizer_group.add_argument(
            "-sgd",
            "--vanilla",
            action="store_const",
            const=self.sgd,
            help="Use Vanilla Stochastic Gradient Descent as optimizer function",
            dest="type_optimizer",
        )
        optimizer_group.add_argument(
            "-mom",
            "--momentum",
            action="store_const",
            const=self.momentum,
            help="Use Momentum as optimizer function",
            dest="type_optimizer",
        )
        optimizer_group.add_argument(
            "-rms",
            "--rms_prop",
            action="store_const",
            const=self.rms_prop,
            help="Use RMSprop as optimizer function",
            dest="type_optimizer",
        )
        optimizer_group.add_argument(
            "-ada",
            "--adam",
            action="store_const",
            const=self.adam,
            help="Use Adam as optimizer function (default)",
            dest="type_optimizer",
        )

    def _activation_arg(self, parser):
        activation_group = parser.add_mutually_exclusive_group(required=False)
        activation_group.add_argument(
            "-sac",
            "--sigmoid",
            action="store_const",
            const={"activation": self.sigmoid, "derivative": self.d_sigmoid},
            help="Use sigmoid as activation function",
            dest="type_activation",
        )
        activation_group.add_argument(
            "-tac",
            "--tanh",
            action="store_const",
            const={"activation": self.tanh, "derivative": self.d_tanh},
            help="Use tanh as activation function (default)",
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
            "-m",
            "--model",
            type=str,
            help=f"Provide dataset NPY file - Using '{self.default_model_file}' as default model file",
            dest="model_file",
        )
        parser.add_argument(
            "-n",
            "--name",
            type=str,
            help=f"Provide name for model saver",
            dest="model_name_file",
        )
        self._activation_arg(parser)
        self._optimizer_arg(parser)
        self._regularization_arg(parser)

    """
    Private Methods
    """

    def _init_args(self):
        self.model_file = self.get_args("model_file")
        self.name = self.get_args("model_name_file", default_value="main")
        if self.model_file is None:
            self.model_file = f"{self.default_model_file}_{self.name}.npy"
        self.activation_func, self.derivative = self.get_args(
            "type_activation",
            default_value={"activation": self.tanh, "derivative": self.d_tanh},
        ).values()
        self.optimizer = self.get_args(
            "type_optimizer",
            default_value=self.adam,
        )
        self.regularization = self.get_args(
            "type_regularization",
            default_value=self.l1_laplacian,
        )

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
                optimizer=self.optimizer,
                regularization=self.regularization,
            )
        )

    def _init_layers(self, input_dim: int, layers_size: int):
        self.input_dim = input_dim
        self.layers_size = layers_size
        for s in layers_size:
            self._add_layer(s)
        self.layers[-1].activation = self.soft_max

    @staticmethod
    @jit(nopython=True)
    def _to_one_hot(y: int, k: int) -> np.array:
        """
        Convertit un entier en vecteur "one-hot".
        to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0)
        """
        one_hot = np.zeros(k)
        one_hot[y] = 1
        return one_hot

    def _feedforward(self, X: np.ndarray):
        self.activations = [X]
        self.weighted_sums = []
        for layer in self.layers:
            self.weighted_sums.append(
                self._weighted_sum(self.activations[-1], layer.weights, layer.biases)
            )
            self.activations.append(layer.activation(self.weighted_sums[-1]))

    def _calcul_backpropagation(self, y: float):
        delta = self.get_output_delta(self.activations[-1], self._to_one_hot(int(y), 2))
        deltas = [delta]

        nb_layers = len(self.layers) - 2
        for i in range(nb_layers, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            activation_prime = layer.activation_prime(self.weighted_sums[i])
            deltas.append(
                self.get_deltas(activation_prime, next_layer.weights, deltas[-1])
            )
        self.deltas = deltas[::-1]

    def _apply_backpropagation(self):
        bias_gradient = []
        weight_gradient = []
        for i in range(len(self.layers)):
            weight_gradient.append(
                self.get_weight_gradient(self.deltas[i], self.activations[i])
            )
            bias_gradient.append(self.deltas[i])
        return weight_gradient, bias_gradient

    def _train_batch(self, X: np.ndarray, Y: np.ndarray, learning_rate: float):
        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]
        for (x, y) in zip(X, Y):
            warnings.simplefilter("default")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._feedforward(x)
            self._calcul_backpropagation(y)
            new_weight_gradient, new_bias_gradient = self._apply_backpropagation()
            for i in range(len(self.layers)):
                weight_gradient[i] += new_weight_gradient[i]
                bias_gradient[i] += new_bias_gradient[i]

        for layer, wg, bg in zip(self.layers, weight_gradient, bias_gradient):
            layer.update_weights(layer.weights, wg / Y.size, learning_rate)
            layer.update_biases(layer.biases, bg / Y.size, learning_rate)

    def _get_loss(self, predicted: np.ndarray, Y: np.ndarray):
        cross_y = np.array([self._to_one_hot(int(y), 2) for y in Y])
        loss = self.cross_entropy(cross_y, np.array(predicted))
        return loss

    """
    Predict
    """

    def _predict_feedforward(self, input_data: np.ndarray):
        activation = input_data
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def _predict(self, input_data: np.ndarray):
        return np.argmax(self._predict_feedforward(input_data))

    def __init__(
        self,
        input_dim: int = None,
        layers_size: list = None,
        epochs: int = 100,
        learning_rate: float = 0.1,
    ):

        self.layers = []
        self.epochs = epochs
        self.learning_rate = learning_rate

        super().__init__()
        self._init_args()
        self.wbdc_preprocess()
        print(
            "\033[92m",
            f"{self.name} init_network {input_dim} {layers_size} {epochs} {learning_rate}",
            "\033[0m",
        )
        if input_dim is not None and layers_size is not None:
            self._init_layers(input_dim, layers_size)

    """
    Public Methods
    """

    def _launch_batch_train(self, X: np.ndarray, Y: np.ndarray, batch_size: int):
        n = Y.size
        for batch_start in range(0, n, batch_size):
            X_batch = X[batch_start : batch_start + batch_size]
            Y_batch = Y[batch_start : batch_start + batch_size]
            self._train_batch(X_batch, Y_batch, self.learning_rate)

    def _check_loss(self, e: int, watch_perf: int):
        if (e > watch_perf or self.loss[-1] < 0.05) and self.best_loss[0] < self.loss[
            -1
        ]:
            self.best_loss = [self.loss[-1], copy.deepcopy(self.layers)]
        return True if np.mean(self.loss[-10:]) < 0.05 else False

    def train(self, batch_size: int = 10):
        logging.info(f"Start training - {self.epochs} epochs")
        watch_perf = int(self.epochs - (self.epochs / 10))
        for e in range(self.epochs):
            start = datetime.datetime.now()

            X_train, Y_train = self.shuffle(self.X, self.Y)
            self._launch_batch_train(X_train, Y_train, batch_size)
            self._evaluate(start, X_train, Y_train, e=e, epochs=self.epochs)
            if self._check_loss(e, watch_perf):
                break

        self.layers = self.best_loss[1] if self.best_loss[1] != 0 else self.layers
        logging.info(f"{self.name} finish")

    def evaluate(self):
        start = datetime.datetime.now()
        X, Y = self.shuffle(self.X_test, self.Y_test)
        self._evaluate_predict(start, X, Y)

    """
    File Handling Method
    """

    def save_model(self, model_file: str):
        model = [(self.input_dim, self.layers_size)]
        model.extend([(l.weights.tolist(), l.biases.tolist()) for l in self.layers])
        logging.info(f"Model saved in '{model_file}'")
        self._save_npy(model_file, model)

    def load_model(self):
        raw_model = self._load_npy(self.model_file)
        input_dim = raw_model[0][0]
        layers_size = [l for l in raw_model[0][1]]
        self._init_layers(input_dim, layers_size)
        for i in range(len(layers_size)):
            self.layers[i].weights = np.array(raw_model[i + 1][0])
            self.layers[i].biases = np.array(raw_model[i + 1][1])
