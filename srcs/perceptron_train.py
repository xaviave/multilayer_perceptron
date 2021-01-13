from tools.Network import Network


def run():
    nn = Network(
        input_dim=30,
        layers_size=[30, 20, 10, 2],
        epochs=20000,
        learning_rate=0.1,
    )
    nn.train()
    nn.evaluate()
    nn.save_model()
    nn._visualize(20000)


if __name__ == "__main__":
    run()
