from tools.Network import Network


def run():
    nn = Network(
        input_dim=30, layers_size=[45, 30, 20, 10, 2], epochs=100, learning_rate=0.5
    )
    nn.train()
    nn.evaluate()
    nn.save_model()


if __name__ == "__main__":
    run()
