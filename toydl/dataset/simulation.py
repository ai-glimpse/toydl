import random

from dataclasses import dataclass


def make_pts(n):
    data = []
    for i in range(n):
        x_1 = random.random()
        x_2 = random.random()
        data.append((x_1, x_2))
    return data


@dataclass
class Graph:
    n: int
    X: list
    y: list


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
