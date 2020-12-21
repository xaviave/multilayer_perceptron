import seaborn as sns
import matplotlib.pyplot as plt

from tools.DataPreprocessing import DataPreprocessing


def visualize(data):
    size_mapping = {"B": 0, "M": 1}
    data.df_dataset["diagnosis"] = data.df_dataset["diagnosis"].map(size_mapping)
    # sns.set_theme(style="ticks")
    # sns.pairplot(data.df_dataset, hue="diagnosis")
    # plt.savefig('line_plot.pdf')
    # # plt.show()


def run():
    data = DataPreprocessing()
    visualize(data)


if __name__ == "__main__":
    run()
