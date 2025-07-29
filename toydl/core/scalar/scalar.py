from __future__ import annotations

import abc
from abc import abstractmethod
from typing import Any, Iterable, Optional, Tuple, Union

import toydl.core.operator as operators

from toydl.core.scalar.bp import backpropagate
from toydl.core.scalar.context import Context
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
        history: ScalarHistory | None = None,
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self._unique_id: int = _var_count
        self.data: float = float(v)
        self.history = history
        self.derivative: Optional[float] = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return (
            f"Scalar(name={self.name}, unique_id={self.unique_id}, data={self.data:.4f}, "
            f"derivative={round(self.derivative, 4) if self.derivative else None})"
        )

    def __mul__(self, b: ScalarLike):
        return Mul.apply(self, b)

    def __rmul__(self, b: ScalarLike):
        return Mul.apply(b, self)

    def __truediv__(self, b: ScalarLike):
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike):
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike):
        return Add.apply(self, b)

    def __bool__(self):
        return bool(self.data)

    def __lt__(self, b: ScalarLike):
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike):
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike):  # type: ignore[override]
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike):
        return Add.apply(self, Neg.apply(b))

    def __rsub__(self, b: ScalarLike):
        return Add.apply(b, Neg.apply(self))

    def __neg__(self):
        return Neg.apply(self)

    def log(self):
        return Log.apply(self)

    def exp(self):
        return Exp.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def relu(self):
        return ReLU.apply(self)

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
        if flag:
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

        derivatives = h.last_fn.backward(h.ctx, d_output)
        variables = h.inputs
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


class ScalarFunction(abc.ABC):
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @staticmethod
    @abstractmethod
    def _forward(*args: Any, **kwargs: Any) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _backward(ctx: Context, d_output: float) -> Union[float, Tuple[float, ...]]:
        raise NotImplementedError

    @classmethod
    def backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        out = cls._backward(ctx, d_out)
        if isinstance(out, tuple):
            return out
        return (out,)

    @classmethod
    def forward(cls, ctx: Context, *inputs: float) -> float:
        return cls._forward(ctx, *inputs)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        raw_vals: list[float] = []
        scalars: list[Scalar] = []
        for v in vals:
            if isinstance(v, Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(Scalar(v))
                raw_vals.append(v)

        # Call forward with the variables.
        ctx = Context()
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, (float, int)), (
            f"Expected return type float or int got {type(c)}"
        )

        # Create a new variable from the result with a new history.
        history = ScalarHistory(cls, ctx, scalars)  # type: ignore
        return Scalar(c, history)


class Add(ScalarFunction):
    """Addition function :math:`f(x, y) = x + y`"""

    @staticmethod
    def _forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    """Log function :math:`f(x) = log(x)`"""

    @staticmethod
    def _forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def _forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> tuple[float, float]:
        a, b = ctx.saved_values
        return operators.mul_back(a, b, d_output)


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def _forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def _forward(ctx: Context, a: float) -> float:
        return operators.neg(a)

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> float:
        return operators.neg_back(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def _forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.sigmoid_back(a, d_output)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def _forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def _forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.exp_back(a, d_output)


class LT(ScalarFunction):
    """Less-than function :math:`f(x) =` 1.0 if x is less than y else 0.0"""

    @staticmethod
    def _forward(ctx: Context, a: float, b: float) -> float:
        return operators.lt(a, b)

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function :math:`f(x) =` 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def _forward(ctx: Context, a: float, b: float) -> float:
        return operators.eq(a, b)

    @staticmethod
    def _backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0
