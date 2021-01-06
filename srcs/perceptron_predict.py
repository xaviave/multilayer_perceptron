from tools.Network import Network


def run():
    print("a refaire ca marche po")
    nn = Network()
    nn.load_model()
    print(nn.evaluate())


if __name__ == "__main__":
    run()
