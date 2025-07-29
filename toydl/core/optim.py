import abc

from typing import Sequence

from toydl.core.module import Parameter


class Optimizer:
    """
    The Optimizer base class
    """

    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters

    @abc.abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self) -> None:
        raise NotImplementedError


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
            if hasattr(p.value, "derivative") and p.value.derivative is not None:
                p.value.derivative = None

    def step(self) -> None:
        """
        Run a sgd step to update parameter value
        """
        for p in self.parameters:
            if hasattr(p.value, "derivative") and p.value.derivative is not None:
                new_value = p.value - self.lr * p.value.derivative
                p.update(new_value)


class Momentum(Optimizer):
    """
    Stochastic Gradient Descent Optimizer

    """

    def __init__(
        self, parameters: Sequence[Parameter], lr: float = 0.01, momentum: float = 0.9
    ):
        """
        Init the SGD optimizer

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
        # Clear delta map
        self.parameter_delta_map = {}

    def step(self) -> None:
        """
        Run a sgd step to update parameter value
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
