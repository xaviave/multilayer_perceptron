import os
import logging
import sys

import numpy as np
import pandas as pd

from tools.args.ArgParser import ArgParser

from tools.args.FileCheckerAction import FileCheckerAction

logging.getLogger().setLevel(logging.INFO)


class DataPreprocessing(ArgParser):
    header = np.array
    df_dataset: pd.DataFrame

    dataset_file: str = None
    dataset_header_file: str = None
    prog_name: str = "Multilayer Perceptron"
    data_path: str = "data"
    model_path: str = os.path.join(data_path, "models")
    dataset_path: str = os.path.join(data_path, "datasets")
    header_path: str = dataset_path
    default_dataset_file: str = os.path.join(dataset_path, "default_dataset.csv")
    default_dataset_header_file: str = os.path.join(
        dataset_path, "default_header_dataset.csv"
    )

    """
    Override Methods
    """

    def _add_exclusive_args(self, parser):
        super()._add_parser_args(parser)
        parser.add_argument(
            "-d",
            "--dataset_file",
            type=str,
            action=FileCheckerAction,
            default=self.default_dataset_file,
            help=f"Provide dataset CSV file - Using '{self.default_dataset_file}' as default file",
        )
        parser.add_argument(
            "-dh",
            "--dataset_header_file",
            type=str,
            action=FileCheckerAction,
            default=self.default_dataset_header_file,
            help=f"Provide dataset header CSV file - Using '{self.default_dataset_header_file}' as default file",
        )

    def _get_options(self):
        self.dataset_file = self.get_args("dataset_file")
        self.dataset_header_file = self.get_args("dataset_header_file")
        if (
            self.dataset_file == self.default_dataset_file
            or self.dataset_header_file == self.default_dataset_header_file
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
        logging.info(f"Reading dataset from file: {file_path}")
        try:
            return pd.read_csv(f"{os.path.abspath(file_path)}")
        except Exception:
            self._handle_error(message=f"Error while processing {file_path}")

    def __init__(self):
        super().__init__(prog=self.prog_name)
        self.df_dataset = self._get_csv_file(self.dataset_file)
        self.header = np.array(self._get_csv_file(self.dataset_header_file).columns)
        self.df_dataset.columns = self.header
        print(self)

    def __str__(self):
        return f"""
{'Summary'.center(70, '=')}
Header file: {self.dataset_header_file}
Dataset file: {self.dataset_file}
Models path: {self.model_path}

{'Dataset stat'.center(70, '=')}
    - header: {self.header}

    - shape: {self.df_dataset.shape}

    - Statistic info:
{self.df_dataset.describe()}
{self.df_dataset.head()}
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
