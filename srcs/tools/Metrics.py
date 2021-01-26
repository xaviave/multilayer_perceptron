import datetime

import warnings
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate
from matplotlib import animation

from tools.Math import Math
from tools.args.ArgParser import ArgParser


class Metrics(ArgParser):
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
        parser.add_argument(
            "-plt",
            "--plot",
            action="store_const",
            const=1,
            help=f"Display animated plot for loss, val_loss, accuracy and val_accuracy",
            dest="plot",
        )

    def __init__(self):
        super().__init__()
        self.verbose = self.get_args("verbose", default_value=0)
        self.plot = self.get_args("plot", default_value=0)

    def visualize(self):
        fig = plt.figure(figsize=(10, 7))
        ep = range(self.last_epoch)
        accuracy = [v * 100 for v in self.accuracy]
        val_accuracy = [v * 100 for v in self.val_accuracy]

        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.xlabel("epochs")
        (lo_1,) = plt.plot([], [], c="r")
        (lo_2,) = plt.plot([], [], c="b")
        plt.ylim(0, max(self.loss) + 0.1)
        plt.xlim(-10, self.last_epoch + 10)
        plt.legend(["training dataset", "validate dataset"])

        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("percents")
        (ac_1,) = plt.plot([], [], c="r")
        (ac_2,) = plt.plot([], [], c="b")
        plt.ylim(min(accuracy) - 1, 100)
        plt.xlim(-10, self.last_epoch + 10)
        plt.legend(["training dataset", "validate dataset"], bbox_to_anchor=(1, 0.12))

        def animate(i):
            lo_1.set_data(ep[:i], self.loss[:i])
            lo_2.set_data(ep[:i], self.val_loss[:i])
            ac_1.set_data(ep[:i], accuracy[:i])
            ac_2.set_data(ep[:i], val_accuracy[:i])
            return (
                lo_1,
                lo_2,
                ac_1,
                ac_2,
            )

        _ = animation.FuncAnimation(
            fig, animate, frames=self.last_epoch, blit=True, interval=100, repeat=False
        )
        plt.show()

    @staticmethod
    def _f1_score(TP: int, FP: int, FN: int):
        try:
            precision = TP / (TP + FP)
            rappel = TP / (TP + FN)
            return 2 / ((1 / precision) + (1 / rappel))
        except ZeroDivisionError:
            return "nan"

    def _get_predictions(self):
        return np.array(list(map(np.argmax, self.activations[-1])))

    def _get_loss(self, X: np.ndarray, Y: np.ndarray):
        self._feedforward(X)
        return self.cross_entropy(self._to_one_hots(Y, 2), self.activations[-1])

    def _get_accuracy(self, Z: np.ndarray, Y: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            warnings.simplefilter(action="ignore", category=DeprecationWarning)
            return np.where(Z == Y)[0].shape[0] / Y.shape[0]

    def additional_metrics(self, predicted: np.ndarray, Y: np.ndarray):
        TP = np.where((Y == predicted) & (Y == 1))[0].shape[0]
        FP = np.where((Y != predicted) & (predicted == 1))[0].shape[0]
        TN = np.where((Y == predicted) & (Y == 0))[0].shape[0]
        FN = np.where((Y != predicted) & (predicted == 0))[0].shape[0]
        print(
            f"F1 score = {round(self._f1_score(TP, FP, FN), 3)} |",
            f"Mean Squared Error = {round(Math.mean_squared(Y, predicted), 3)}\n{'-'*70}\n",
            "Confusion Matrix\n",
            tabulate(
                [["False", TN, FP], ["True", FN, TP]],
                headers=[f"sample size={Y.shape[0]}", "False", "True"],
            ),
            f"\n{'-'*70}",
        )

    def _evaluate(self, start: int, X: np.ndarray, Y: np.ndarray, e: int, epochs: int):
        self.loss.append(self._get_loss(X, Y))
        Z = self._get_predictions()
        self.accuracy.append(self._get_accuracy(Z, Y))
        self.val_loss.append(self._get_loss(self.X_val, self.Y_val))
        Z_val = self._get_predictions()
        self.val_accuracy.append(self._get_accuracy(Z_val, self.Y_val))
        if self.verbose:
            self.additional_metrics(Z, Y)
        time = datetime.datetime.now() - start
        print(
            f"epoch {e + 1}/{epochs} - loss: {self.loss[-1]:.4f} - accuracy: {self.accuracy[-1] * 100:.2f}% - val_loss {self.val_loss[-1]:.4f} - val_accuracy: {self.val_accuracy[-1] * 100:.2f}% - time: {time}"
        )

    def _evaluate_predict(self, start: int, X: np.ndarray, Y: np.ndarray):
        loss = self._get_loss(X, Y)
        Z = self._get_predictions()
        accuracy = self._get_accuracy(Z, Y)
        time = datetime.datetime.now() - start
        if self.verbose:
            self.additional_metrics(Z, Y)
        print(
            f"Predict model | loss: {loss:.4f} - accuracy: {accuracy * 100:.2f}% - time: {time}"
        )
