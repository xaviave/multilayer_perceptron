from tools.PerceptronGenetic import PerceptronGenetic


def launch():
    gen = PerceptronGenetic(
        20, mutation=1, timer=3000, loss=0.07, input_size=30, output_size=2
    )
    gen.launch()


if __name__ == "__main__":
    launch()
