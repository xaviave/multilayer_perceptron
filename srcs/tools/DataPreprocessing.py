import datetime
import os
import logging
import sys

import numpy as np
import pandas as pd

from tools.Math import Math
from tools.args.ArgParser import ArgParser

from tools.args.FileCheckerAction import FileCheckerAction

logging.getLogger().setLevel(logging.INFO)


class DataPreprocessing(ArgParser, Math):
    X: np.ndarray
    Y: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    df_dataset_test: pd.DataFrame
    df_dataset_train: pd.DataFrame

    data_path: str = "data"
    dataset_test_file: str = None
    dataset_train_file: str = None
    prog_name: str = "Multilayer Perceptron"
    model_path: str = os.path.join(data_path, "models")
    dataset_path: str = os.path.join(data_path, "datasets")
    default_dataset_file: str = os.path.join(dataset_path, "default_dataset.csv")

    """
    Override Methods
    """

    def _add_parser_args(self, parser):
        super()._add_parser_args(parser)
        parser.add_argument(
            "-d",
            "--dataset_train_file",
            type=str,
            action=FileCheckerAction,
            default=self.default_dataset_file,
            help=f"Provide dataset CSV file - Using '{self.default_dataset_file}' as default file",
        )
        parser.add_argument(
            "-dt",
            "--dataset_test_file",
            type=str,
            action=FileCheckerAction,
            default=self.default_dataset_file,
            help=f"Provide dataset CSV file - Using '{self.default_dataset_file}' as default test file",
        )
        parser.add_argument(
            "-s",
            "--split_train",
            type=int,
            default=0,
            help=f"Split train dataset by X to create a test dataset\nIf provided, override '-dt' option",
        )

    def _get_options(self):
        self.split = self.get_args("split_train")
        self.dataset_test_file = self.get_args("dataset_test_file")
        self.dataset_train_file = self.get_args("dataset_train_file")
        if (
            self.dataset_test_file == self.default_dataset_file
            or self.dataset_train_file == self.default_dataset_file
        ):
            logging.info("Using default dataset CSV file")

    """
    Private Methods
    """

    @staticmethod
    def _handle_error(exception=None, message="Error", mod=-1):
        if exception:
            logging.error(f"{exception}")
        logging.error(f"{message}")
        sys.exit(mod)

    def _load_npy(self, file_name: str):
        try:
            return np.load(file_name, allow_pickle=True)
        except Exception as e:
            self._handle_error(exception=e)

    def _save_npy(self, file_name: str, data):
        try:
            np.save(file_name, data, allow_pickle=True)
        except Exception as e:
            self._handle_error(exception=e)

    def _get_csv_file(self, file_path: str) -> pd.DataFrame:
        start = datetime.datetime.now()
        logging.info(f"Reading dataset from file: {file_path}")
        try:
            return pd.read_csv(f"{os.path.abspath(file_path)}", header=None)
        except Exception:
            self._handle_error(message=f"Error while processing {file_path}")
        logging.info(f"data loaded: {datetime.datetime.now() - start}")

    def __init__(self):
        super().__init__(prog=self.prog_name)
        self.df_dataset_train = self._get_csv_file(self.dataset_train_file)
        if 0 < self.split < 100:
            self.split = int(len(self.df_dataset_train.index) * (self.split / 100))
            self.df_dataset_test = self.df_dataset_train[self.split :]
            self.df_dataset_train = self.df_dataset_train[: self.split]
        else:
            self.df_dataset_test = self._get_csv_file(self.dataset_test_file)

    def __str__(self):
        return f"""
{'Summary'.center(70, '=')}
Dataset train file: {self.dataset_train_file}
Dataset test file: {self.dataset_test_file}
Models path: {self.model_path}

{'Dataset stat'.center(70, '=')}
    - shape: {self.df_dataset_train.shape}

    - Statistic info:
{self.df_dataset_train.describe()}
{self.df_dataset_train.head()}
{'=' * 70}
"""

    """
    Public Methods
    """

    def write_to_csv(self, file_name: str, dataset: list, columns: list):
        tmp_dataset = pd.DataFrame(data=dataset)
        tmp_dataset.index.name = "Index"
        tmp_dataset.columns = columns
        try:
            with open(os.path.abspath(file_name), "w") as file:
                file.write(tmp_dataset.to_csv())
        except Exception as e:
            self._handle_error(exception=e)

    @staticmethod
    def df_to_np(df):
        y = df.pop(0)
        return df.to_numpy() / 255, y

    def wbdc_preprocess(self):
        self.Y = np.where(self.df_dataset_train.pop(1) == "M", 0, 1)
        del self.df_dataset_train[0]
        self.X = self.normalize(self.df_dataset_train.to_numpy())
        self.X_test = self.X
        self.Y_test = self.Y
