from tools.Network import Network


def run():
    nn = Network(input_dim=784, layers_size=[200, 10])
    nn.train(epochs=1, learning_rate=1)
    nn.save_model()

    print(nn.evaluate())
    nn.layers = []
    nn.load_model()
    print(nn.evaluate())


if __name__ == "__main__":
    run()
