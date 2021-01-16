from tools.Network import Network


def run():
    epochs = 1000
    nn = Network(
        input_dim=30,
        layers_size=[20, 10, 2],
        epochs=epochs,
        learning_rate=0.01,
    )
    nn.train()
    nn.evaluate()
    nn.save_model(nn.model_file)
    # nn._visualize(epochs)


if __name__ == "__main__":
    run()
