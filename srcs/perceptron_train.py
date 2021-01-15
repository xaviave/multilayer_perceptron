from tools.Network import Network


def run():
    epochs = 10000
    nn = Network(
        input_dim=30,
        layers_size=[20, 18, 16, 14, 10, 2],
        epochs=epochs,
        learning_rate=0.08,
    )
    nn.train()
    nn.evaluate()
    nn.save_model(nn.model_file)
    # nn._visualize(epochs)


if __name__ == "__main__":
    run()
