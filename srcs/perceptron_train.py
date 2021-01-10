from tools.Network import Network


def run():
    nn = Network(input_dim=30, layers_size=[40, 40, 30,30,20, 20,10, 10, 2], epochs=100000, learning_rate=0.001)
    nn.train()
    nn.evaluate()
    nn.save_model()


if __name__ == "__main__":
    run()
