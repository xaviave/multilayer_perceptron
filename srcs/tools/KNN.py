import math
import sys

import numpy as np


class KNN:
    @staticmethod
    def _euclidean_distance(point1, point2):
        return np.sqrt(np.sum(np.power(point1 - point2, 2)))
        # sum_squared_distance = 0
        # for i in range(len(point1)):
        #     sum_squared_distance += math.pow(point1[i] - point2[i], 2)
        # return math.sqrt(sum_squared_distance)

    def _knn(self, dataset, query, k, distance_fn):
        neighbor_distances_and_indices = np.array(
            [(i, distance_fn(d, query)) for i, d in enumerate(dataset)]
        )
        return [
            int(r[0])
            for r in sorted(neighbor_distances_and_indices, key=lambda x: x[1])[:k]
        ]

    # numpy.linalg.norm(a - b) eculidan_distance
    def knn_imputer(self, dataset, column, queries, k):
        for i, q in enumerate(queries):
            tmp = np.delete(q.copy(), column)
            indexes = self._knn(
                dataset=dataset,
                query=tmp,
                k=k,
                distance_fn=self._euclidean_distance,
            )
            selected = dataset[indexes]
            queries[i][column] = np.mean(selected[:, column])
        return queries

    def knn_multi_column_imputer(self, dataset, k):
        np.set_printoptions(threshold=sys.maxsize)
        dataset = np.delete(dataset, 0, 1)
        dataset[:, 0] = np.where(dataset[:, 0] == "M", -1, -2)
        good = dataset[(dataset != 0).all(axis=-1)]
        for i in range(dataset.shape[1]):
            c = dataset[:, i] == 0
            if len(dataset[c]) > 0:
                queries = dataset[c]
                up = self.knn_imputer(np.delete(good.copy(), i, 1), i, queries, k)
                dataset[c] = up
        return np.array(dataset, dtype=float)
