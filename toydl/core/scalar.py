from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

from toydl.core.autodiff import Context, Variable, backpropagate
import toydl.core.operator as operators

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """
    `ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


_var_count = 0


class Scalar:
    """
    A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        return Mul.apply(self, b)

    def __truediv__(self, b):
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b):
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b):
        return Add.apply(self, b)

    def __bool__(self):
        return bool(self.data)

    def __lt__(self, b):
        return LT.apply(self, b)

    def __gt__(self, b):
        return LT.apply(b, self)

    def __eq__(self, b):
        return EQ.apply(self, b)

    def __sub__(self, b):
        return Add.apply(self, Neg.apply(b))

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

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated
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
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        derivatives = h.last_fn._backward(h.ctx, d_output)
        derivatives = derivatives if isinstance(derivatives, Iterable) else (derivatives,)
        variables = h.inputs if isinstance(h.inputs, Iterable) else (h.inputs,)
        var_derivatives = [(var, derivative) for (var, derivative) in zip(variables, derivatives)
                           if not var.is_constant()]
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


def wrap_tuple(x):  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    """Turn a singleton tuple into a value"""
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, (float, int)), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = ScalarHistory(cls, ctx, scalars)
        return Scalar(c, back)


class Add(ScalarFunction):
    """Addition function :math:`f(x, y) = x + y`"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    """Log function :math:`f(x) = log(x)`"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        a, b = ctx.saved_values
        return operators.mul_back(a, b, d_output)


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.neg_back(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.sigmoid_back(a, d_output)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values[0]
        return operators.exp_back(a, d_output)


class LT(ScalarFunction):
    """Less-than function :math:`f(x) =` 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function :math:`f(x) =` 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0
