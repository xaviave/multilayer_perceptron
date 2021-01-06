from tools.Network import Network


def run():
    nn = Network(input_dim=30, layers_size=[45, 20, 2])
    # nn.train(epochs=22, learning_rate=1) -> 0.0 loss | 100 accuracy
    nn.train(epochs=100, learning_rate=1)
    nn.save_model()

    print(nn.evaluate())
    nn.layers = []
    nn.load_model()
    print(nn.evaluate())


if __name__ == "__main__":
    run()
