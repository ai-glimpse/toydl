import random

from typing import List

from toydl.core.module import Module
from toydl.core.scalar import Scalar


class Linear(Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(f"bias_{j}", Scalar(2 * (random.random() - 0.5)))
            )

    def forward(self, inputs: List[Scalar]) -> List[Scalar]:
        outputs = []
        n, m = len(self.weights), len(self.weights[0])
        for j in range(m):
            output = Scalar(0)
            for i in range(n):
                output += self.weights[i][j].value * inputs[i] + self.bias[j].value
            outputs.append(output)
        return outputs
