from typing import Sequence

from toydl.core.module import Parameter


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None

    def step(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue
            else:
                if hasattr(p.value, "derivative") and p.value.derivative is not None:
                    # print(f"{p.name}: {self.lr * p.value.derivative}")
                    new_value = p.value - self.lr * p.value.derivative
                    p.update(new_value)
                    # p.update(p.value)
