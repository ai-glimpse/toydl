from ._base import Optimizer

from typing import Sequence

from toydl.core.module import Parameter


class Momentum(Optimizer):
    """
    Momentum Optimizer
    """

    def __init__(
        self, parameters: Sequence[Parameter], lr: float = 0.01, momentum: float = 0.9
    ):
        """
        Init the Momentum optimizer

        :param parameters: the parameters that will be optimized
        :param lr: learning rate
        :param momentum: momentum coefficient
        """
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.parameter_delta_map: dict[Parameter, float] = {}

    def zero_grad(self) -> None:
        """
        Clear the grad/derivative value of parameter
        """
        for p in self.parameters:
            if hasattr(p.value, "derivative") and p.value.derivative is not None:
                p.value.derivative = None

    def step(self) -> None:
        """
        Run a momentum step to update parameter value
        """
        for p in self.parameters:
            if hasattr(p.value, "derivative") and p.value.derivative is not None:
                delta = (
                    -self.lr * p.value.derivative
                    + self.momentum * self.parameter_delta_map.get(p, 0)
                )
                self.parameter_delta_map[p] = delta
                new_value = p.value + delta
                p.update(new_value)
