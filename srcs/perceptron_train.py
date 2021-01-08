from tools.Network import Network


def run():
    nn = Network(input_dim=30, layers_size=[20, 10, 2])
    nn.train(epochs=100, learning_rate=0.1)
    nn.evaluate()
    nn.save_model()


if __name__ == "__main__":
    run()
