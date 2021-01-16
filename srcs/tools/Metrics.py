import datetime

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

    def __init__(self):
        super().__init__()
        self.verbose = self.get_args("verbose", default_value=0)

    def visualizer(self, epochs: int, val_loss: float, loss: float):
        fig = plt.figure(figsize=(10, 7))
        ep = range(epochs)
        val_loss = [v * 100 for v in val_loss]

        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.xlabel("epochs")
        (line,) = plt.plot([], [], c="r")
        plt.ylim(min(loss) - 0.2, max(loss) + 0.2)
        plt.xlim(-10, epochs + 10)

        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("percents")
        (line2,) = plt.plot([], [], c="b")
        plt.ylim(min(val_loss) - 5, max(val_loss) + 5)
        plt.xlim(-10, epochs + 10)

        def animate(i: int, loss: float, val_loss: float):
            line.set_data(ep[:i], loss[:i])
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
    def _f1_score(TP: int, FP: int, FN: int):
        try:
            precision = TP / (TP + FP)
            rappel = TP / (TP + FN)
            return 2 / ((1 / precision) + (1 / rappel))
        except ZeroDivisionError:
            return "nan"

    def additional_metrics(self, predicted: np.ndarray, Y: np.ndarray):
        TP = np.where((Y == predicted) & (Y == 1))[0].shape[0]
        FP = np.where((Y != predicted) & (predicted == 1))[0].shape[0]
        TN = np.where((Y == predicted) & (Y == 0))[0].shape[0]
        FN = np.where((Y != predicted) & (predicted == 0))[0].shape[0]
        print(
            f"F1 score = {self._f1_score(TP, FP, FN)}",
            f"Mean squared Error = {Math.mean_squared(Y, predicted)}\n",
            "Confusion Matrix\n",
            tabulate(
                [["False", TN, FP], ["True", FN, TP]],
                headers=[f"sample size={Y.shape[0]}", "False", "True"],
            ),
            f"\n{'-'*70}",
        )

    def _evaluate(self, start: int, X: np.ndarray, Y: np.ndarray, e: int, epochs: int):
        self.loss.append(
            self._get_loss(np.array([self._predict_feedforward(x) for x in X]), Y)
        )
        self.val_loss.append(
            self._get_loss(
                [self._predict_feedforward(x) for x in self.X_val], self.Y_val
            )
        )
        time = datetime.datetime.now() - start
        if self.verbose:
            self.additional_metrics(np.array([self._predict(x) for x in X]), Y)
        print(
            f"""
epoch {e + 1}/{epochs} - loss: {self.loss[-1]:.4f} - val_loss {self.val_loss[-1]:.4f} - time: {time}
"""
        )

    def _evaluate_predict(self, start: int, X: np.ndarray, Y: np.ndarray):
        loss = self._get_loss(np.array([self._predict_feedforward(x) for x in X]), Y)
        time = datetime.datetime.now() - start
        if self.verbose:
            self.additional_metrics(np.array([self._predict(x) for x in X]), Y)
        print(f"Predict model | loss: {loss:.4f}  - acc {'loic'} - time: {time}")
