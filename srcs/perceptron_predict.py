from tools.Network import Network


def run():
    nn = Network()
    nn.load_model()
    nn.evaluate()


if __name__ == "__main__":
    run()
