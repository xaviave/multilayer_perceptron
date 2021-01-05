import logging
import datetime

import numpy as np
import pandas as pd

from sklearn.utils import shuffle

from tools.Network import Network


def load_mnist_data(path: str) -> tuple:
    train_data = pd.read_csv(path, header=None)
    np.random.shuffle(train_data.values)
    labels = train_data.pop(0)
    imgs = train_data.to_numpy() / 255
    # imgs = imgs.reshape(imgs.shape[0], 28, 28)
    return imgs, labels


if __name__ == "__main__":
    start = datetime.datetime.now()
    data_path = "data/datasets/mnist/"
    X_train, Y_train = load_mnist_data(data_path + "mnist_train.csv")
    X_test, Y_test = load_mnist_data(data_path + "mnist_test.csv")
    logging.warning(f"data loaded: {datetime.datetime.now() - start}")
    net = Network(input_dim=784, layers_size=[200, 10])
    tps = []
    perfs = []
    for i in range(1):
        start = datetime.datetime.now()
        net.train(X_train, Y_train, epochs=5, learning_rate=1)
        accuracy = net.evaluate(X_test, Y_test)
        perfs.append(accuracy)
        tps.append(datetime.datetime.now() - start)
        logging.warning(f"New Perf: {perfs[-1] * 100}% | time: {net.times[-1]}")
    logging.warning(
        f"Mean perf : {np.mean(perfs) * 100}% | mean time: {np.mean(net.times)}"
    )
