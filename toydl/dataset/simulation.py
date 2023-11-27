import random

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Graph:
    n: int
    X: list
    y: list

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index) -> "Graph":
        if isinstance(index, slice):
            xs = self.X[index]
            ys = self.y[index]
            return Graph(len(xs), xs, ys)
        elif isinstance(index, int):
            return Graph(1, self.X[index], self.y[index])
        else:
            raise TypeError(f"Invalid argument type: {type(index)}, {index}")

    def train_test_split(
        self, train_proportion: float = 0.7
    ) -> Tuple["Graph", "Graph"]:
        training_size = int(len(self) * train_proportion)
        training_set = self[:training_size]
        test_set = self[training_size:]
        return training_set, test_set


def make_pts(n):
    data = []
    for i in range(n):
        x_1 = random.random()
        x_2 = random.random()
        data.append((x_1, x_2))
    return data


def simple(n):
    data = make_pts(n)
    y = []
    for x_1, x_2 in data:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(n, data, y)


def diag(n):
    data = make_pts(n)
    y = []
    for x_1, x_2 in data:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(n, data, y)


def split(n):
    data = make_pts(n)
    y = []
    for x_1, x_2 in data:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(n, data, y)


def xor(n):
    data = make_pts(n)
    y = []
    for x_1, x_2 in data:
        y1 = 1 if ((x_1 < 0.5 < x_2) or (x_1 > 0.5 > x_2)) else 0
        y.append(y1)
    return Graph(n, data, y)


datasets = {"Simple": simple, "Diag": diag, "Split": split, "Xor": xor}
