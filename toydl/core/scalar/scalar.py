from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple, Union

import toydl.core.scalar.function as fn

from toydl.core.scalar.bp import backpropagate
from toydl.core.scalar.history import ScalarHistory

ScalarLike = Union[float, int, "Scalar"]


_var_count: int = 0


class Scalar:
    """
    A reimplementation of scalar values for auto-differentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self._unique_id: int = _var_count
        self.data: float = float(v)
        self.history: ScalarHistory = back
        self.derivative: Optional[float] = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        return fn.Mul.apply(self, b)

    def __rmul__(self, b):
        return fn.Mul.apply(b, self)

    def __truediv__(self, b):
        return fn.Mul.apply(self, fn.Inv.apply(b))

    def __rtruediv__(self, b):
        return fn.Mul.apply(b, fn.Inv.apply(self))

    def __add__(self, b):
        return fn.Add.apply(self, b)

    def __bool__(self):
        return bool(self.data)

    def __lt__(self, b):
        return fn.LT.apply(self, b)

    def __gt__(self, b):
        return fn.LT.apply(b, self)

    def __eq__(self, b):
        return fn.EQ.apply(self, b)

    def __sub__(self, b):
        return fn.Add.apply(self, fn.Neg.apply(b))

    def __rsub__(self, b):
        return fn.Add.apply(b, fn.Neg.apply(self))

    def __neg__(self):
        return fn.Neg.apply(self)

    def log(self):
        return fn.Log.apply(self)

    def exp(self):
        return fn.Exp.apply(self)

    def sigmoid(self):
        return fn.Sigmoid.apply(self)

    def relu(self):
        return fn.ReLU.apply(self)

    @property
    def unique_id(self):
        return self._unique_id

    def requires_grad_(self, flag: bool = True):
        """
        Set the requires_grad flag to `flag` on variable.

        Ensures that operations on this variable will trigger
        backpropagation.

        :param flag: whether to require grad
        """
        self.history = ScalarHistory()

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `x` to the derivative accumulated on this variable.
        Should only be called during auto-differentiation on leaf variables.

        :param x: value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Scalar]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Scalar, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        derivatives = h.last_fn._backward(h.ctx, d_output)
        derivatives = (
            derivatives if isinstance(derivatives, Iterable) else (derivatives,)
        )
        variables = h.inputs if isinstance(h.inputs, Iterable) else (h.inputs,)
        var_derivatives = [
            (var, derivative)
            for (var, derivative) in zip(variables, derivatives)
            if not var.is_constant()
        ]
        return var_derivatives

    def backward(self, d_output: Optional[float] = None) -> None:
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)
