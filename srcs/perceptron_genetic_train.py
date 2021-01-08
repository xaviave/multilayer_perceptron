from tools.PerceptronGenetic import PerceptronGenetic


def launch():
    gen = PerceptronGenetic(
        10, mutation=1, timer=30, loss=0.07, input_size=30, output_size=2
    )
    gen.launch()


if __name__ == "__main__":
    launch()
