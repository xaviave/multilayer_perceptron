import datetime
import os
import logging
import sys
import warnings

import numpy as np
import pandas as pd

from tools.KNN import KNN
from tools.Math import Math
from tools.args.ArgParser import ArgParser

from tools.args.FileCheckerAction import FileCheckerAction


logging.getLogger().setLevel(logging.INFO)


class DataPreprocessing(ArgParser, Math, KNN):
    X: np.ndarray
    Y: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
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

    def _scaler_arg(self, parser):
        scaler_group = parser.add_mutually_exclusive_group(required=False)
        scaler_group.add_argument(
            "-nsc",
            "--normalize",
            action="store_const",
            const=self.normalize,
            help="Use Normalization as scaler function",
            dest="type_scaler",
        )
        scaler_group.add_argument(
            "-ssc",
            "--standardize",
            action="store_const",
            const=self.standardize,
            help="Use Standardization as scaler function (default)",
            dest="type_scaler",
        )

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
            "-sp",
            "--split_train",
            type=int,
            default=0.3,
            help=f"Split train dataset by X to create a test dataset\nIf provided, override '-dt' option",
        )
        self._scaler_arg(parser)

    def _get_options(self):
        self.split = self.get_args("split_train")
        self.dataset_test_file = self.get_args("dataset_test_file")
        self.dataset_train_file = self.get_args("dataset_train_file")
        self.scaler = self.get_args("type_scaler", default_value=self.standardize)
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
            warnings.simplefilter("default")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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

    @staticmethod
    def _fix_data(df):
        X = df.to_numpy()
        X_ = X[(X != 0).all(axis=-1)]
        for i in range(X.shape[1]):
            m = np.mean(X_[i])
            X[X[:, i] == 0] = m
        return X

    @staticmethod
    def _create_validation_dataset(split, X, Y):
        i = int(X.shape[0] - X.shape[0] * split)
        return Y[:i], X[:i], Y[i:], X[i:]

    def _split_dataset(self, dataset):
        np.set_printoptions(threshold=sys.maxsize)
        dataset = np.delete(dataset, 0, axis=1)
        dataset[dataset == 0] = np.nan
        dataset[:, 0] = np.where(dataset[:, 0] == "M", 0, 1)
        X = self.knn_multi_column_imputer(np.array(dataset, dtype=float), 2)
        return X[:, 0], self.scaler(X[:, 1:])

    def __init__(self):
        super().__init__(prog=self.prog_name)
        self.df_dataset_train = self._get_csv_file(self.dataset_train_file)
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

    @staticmethod
    def shuffle(X, Y):
        c = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
        np.random.shuffle(c)
        return (
            c[:, : X.size // len(X)].reshape(X.shape),
            c[:, X.size // len(X) :].reshape(Y.shape),
        )

    def write_to_csv(self, file_name: str, dataset: list, columns: list):
        tmp_dataset = pd.DataFrame(data=dataset)
        tmp_dataset.index.name = "Index"
        tmp_dataset.columns = columns
        try:
            with open(os.path.abspath(file_name), "w") as file:
                file.write(tmp_dataset.to_csv())
        except Exception as e:
            self._handle_error(exception=e)

    def wbdc_preprocess(self):
        self.Y, self.X = self._split_dataset(self.df_dataset_train.to_numpy())
        self.Y_test, self.X_test = self._split_dataset(self.df_dataset_test.to_numpy())
        self.Y, self.X, self.Y_val, self.X_val = self._create_validation_dataset(
            self.split, *self.shuffle(self.X, self.Y)
        )
