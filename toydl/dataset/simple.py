from dataclasses import dataclass
from typing import List, Tuple, Union

import matplotlib.pyplot as plt

from toydl.dataset.base import DataSetBase
from toydl.dataset.exception import DatasetValidationError


@dataclass(repr=True)
class SimpleDataset(DataSetBase):
    xs: List[List[Union[int, float]]]
    ys: List[int]

    def __post_init__(self):
        if len(self.xs) != len(self.ys):
            raise DatasetValidationError(
                f"X, y lengths don't match, "
                f"lengths: X({len(self.xs)}), y({len(self.ys)})"
            )

        item_lengths = [len(xi) for xi in self.xs]
        if len(set(item_lengths)) != 1:
            raise DatasetValidationError(
                f"X items have different lengths: {set(item_lengths)}"
            )

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, index) -> "SimpleDataset":
        if isinstance(index, slice):
            return SimpleDataset(self.xs[index], self.ys[index])
        elif isinstance(index, int):
            return SimpleDataset([self.xs[index]], [self.ys[index]])
        else:
            raise TypeError(f"Invalid argument type: {type(index)}, {index}")

    def __iter__(self):
        return iter(zip(self.xs, self.ys))

    def train_test_split(
        self, train_proportion: float = 0.7
    ) -> Tuple["SimpleDataset", "SimpleDataset"]:
        training_size = int(len(self) * train_proportion)
        training_set = self[:training_size]
        test_set = self[training_size:]
        return training_set, test_set

    def plot(self, filename: str = "dataset.png"):
        x_values = [x[0] for x in self.xs]
        y_values = [x[1] for x in self.xs]

        # Assign colors based on the y values
        colors = ["green" if y == 0 else "red" for y in self.ys]

        # Plot the data points with specified colors
        plt.scatter(x_values, y_values, c=colors)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        # plt.show()


if __name__ == "__main__":
    data = SimpleDataset([[1, 1], [2, 2]], [1, 2])
    data.plot()
