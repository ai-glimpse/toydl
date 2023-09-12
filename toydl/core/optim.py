from typing import Sequence

from toydl.core.module import Parameter


class Optimizer:
    """
    The Optimizer base class
    """

    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    """
    Stochastic Gradient Descent Optimizer

    """

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """
        Init the SGD optimizer

        :param parameters: the parameters that will be optimized
        :param lr: learning rate
        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """
        Clear the grad/derivative value of parameter
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None

    def step(self) -> None:
        """
        Run a sgd step to update parameter value
        """
        for p in self.parameters:
            if p.value is None:
                continue
            else:
                if hasattr(p.value, "derivative") and p.value.derivative is not None:
                    new_value = p.value - self.lr * p.value.derivative
                    p.update(new_value)
