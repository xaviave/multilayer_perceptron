import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from tabulate import tabulate

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

    def visualizer(self, epochs, val_loss, loss):
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

        def animate(i, loss, val_loss):
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
    def _f1_score(TP, FP, FN):
        if TP + FP == 0 or TP + FN == 0:
            return 'nan'
        precision = TP / (TP + FP)
        rappel = TP / (TP + FN)
        return 2 / ((1 / precision) + (1 / rappel))

    def additional_metrics(self, predicted, Y):
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
