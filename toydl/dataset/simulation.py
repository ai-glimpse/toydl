import random

from typing import List

from toydl.dataset.simple import SimpleDataset


def generate_random_points(n: int) -> List[List[float]]:
    points = [[random.random(), random.random()] for _ in range(n)]
    return points


def simple(n: int) -> SimpleDataset:
    xs = generate_random_points(n)
    ys = []
    for x1, x2 in xs:
        y = 1 if x1 < 0.5 else 0
        ys.append(y)
    return SimpleDataset(xs, ys)


def diag(n: int) -> SimpleDataset:
    xs = generate_random_points(n)
    ys = []
    for x1, x2 in xs:
        y1 = 1 if x1 + x2 < 0.5 else 0
        ys.append(y1)
    return SimpleDataset(xs, ys)


def split(n: int) -> SimpleDataset:
    xs = generate_random_points(n)
    ys = []
    for x1, x2 in xs:
        y = 1 if x1 < 0.2 or x1 > 0.8 else 0
        ys.append(y)
    return SimpleDataset(xs, ys)


def xor(n: int) -> SimpleDataset:
    xs = generate_random_points(n)
    ys = []
    for x1, x2 in xs:
        y = 1 if ((x1 < 0.5 < x2) or (x1 > 0.5 > x2)) else 0
        ys.append(y)
    return SimpleDataset(xs, ys)
