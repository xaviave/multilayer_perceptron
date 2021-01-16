import math
import sys

import numpy as np


class KNN:
    @staticmethod
    def arange_array(dataset, clear_dt: np.ndarray, columns: np.ndarray):
        store = np.zeros(shape=(dataset.shape[0], dataset.shape[1] - clear_dt.shape[1]))
        for i, c in enumerate(columns):
            store[:, i] = dataset[:, c]
        return np.array(
            np.c_[clear_dt.reshape(len(clear_dt), -1), store.reshape(len(store), -1)],
            dtype=float,
        )

    @staticmethod
    def rearange_array(
        dataset: np.ndarray, initial_colomns: np.ndarray, columns: np.ndarray
    ):
        index = list(range(dataset.shape[1] - len(columns)))
        for i, c in zip(initial_colomns, columns):
            index.insert(i, c)
        new_dataset = np.zeros(dataset.shape)
        for i, c in enumerate(index):
            new_dataset[:, i] = dataset[:, c]
        return np.array(new_dataset, dtype=float)

    @staticmethod
    def _euclidean_distance(point1: np.ndarray, point2: np.ndarray):
        return np.sqrt(np.sum(np.power(point1 - point2, 2)))

    def _knn(self, dataset: np.ndarray, query: np.ndarray, k: int, distance_fn):
        neighbor_distances_and_indices = np.array(
            [(i, distance_fn(d, query)) for i, d in enumerate(dataset)]
        )
        return [
            int(r[0])
            for r in sorted(neighbor_distances_and_indices, key=lambda x: x[1])[:k]
        ]

    def knn_imputer(
        self,
        dataset: np.ndarray,
        column: np.ndarray,
        queries: np.ndarray,
        k: int,
        to_fill: np.ndarray,
        rd: np.ndarray,
    ):
        for i, q in enumerate(queries):
            indexes = self._knn(
                dataset=dataset,
                query=q,
                k=k,
                distance_fn=self._euclidean_distance,
            )
            selected = rd[indexes]
            to_fill[i][column] = np.mean(selected[:, column])
        return to_fill

    def knn_multi_column_imputer(self, dataset: np.ndarray, k: np.ndarray):
        initial_columns = np.unique(np.argwhere(np.isnan(dataset)).T[1])
        dataset = self.arange_array(
            dataset,
            np.ma.compress_cols(np.ma.masked_invalid(dataset.copy())),
            initial_columns,
        )
        indexes = np.unique(np.argwhere(np.isnan(dataset)).T[0])
        columns = np.unique(np.argwhere(np.isnan(dataset)).T[1])
        queries = dataset[np.isnan(dataset).any(axis=1)]
        good = dataset[~np.isnan(dataset).any(axis=1)]
        ds = np.ma.compress_cols(np.ma.masked_invalid(good[:, :-6].copy()))
        tmp_queries = np.ma.compress_cols(np.ma.masked_invalid(queries.copy()))
        for i in columns:
            queries = self.knn_imputer(ds, i, tmp_queries, k, queries, good)
        for i, q in enumerate(indexes):
            dataset[q] = queries[i]
        return self.rearange_array(dataset, initial_columns, columns)
