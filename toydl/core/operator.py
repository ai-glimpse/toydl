import math


def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y


def mul_back(x: float, y: float, d: float):
    return d * y, d * x


def id_(x: float) -> float:
    """$f(x) = x$"""
    return x


def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y


def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x


def neg_back(d: float) -> float:
    return -d


def lt(x: float, y: float) -> float:
    """$f(x, y) = x < y$"""
    return float(x < y)


def eq(x: float, y: float) -> float:
    return float(abs(x - y) <= 1e-8)


def max_(x, y):
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return float(abs(x - y) < 1e-2)


def sigmoid(x: float) -> float:
    r"""
    $$f(x) =  \frac{1.0}{(1.0 + e^{-x})}$$

    Calculate as

    $$
    f(x) = \begin{cases}
    \frac{1.0}{1.0 + e^{-x}} & \text{if } x \geq 0 \\
    \frac{e^x}{1.0 + e^x} & \text{otherwise}
    \end{cases}
    $$

    for stability.
    The key is to make sure the `x` in exp(x) is always negative to avoid exp(x) overflow.
    """
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def sigmoid_back(x: float, d: float) -> float:
    return d * sigmoid(x) * (1 - sigmoid(x))


def relu(x: float) -> float:
    """
    f(x) = x if x is greater than 0, else 0
    """
    return x if x > 0 else 0


def log(x: float) -> float:
    return math.log(x + 1e-6)


def exp(x: float) -> float:
    return math.exp(x)


def exp_back(x: float, d: float) -> float:
    return d * exp(x)


def log_back(x: float, d: float) -> float:
    return d / (x + 1e-6)


def inv(x: float) -> float:
    return 1 / x


def inv_back(x: float, d: float) -> float:
    return -d / x**2


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0
