import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y


def mul_back(x: float, y: float, d: float):
    return d * y, d * x


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def neg_back(d: float) -> float:
    return -d


def lt(x: float, y: float):
    return float(x < y)


def eq(x: float, y: float) -> float:
    return float(abs(x - y) <= 1e-8)


def max(x, y):
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return float(abs(x - y) < 1e-2)


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}$
    """
    # IF-ELSE is used to make sure the `x` in exp(x) is always negative to avoid exp(x) overflow
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


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def __f(items):
        return [fn(item) for item in items]

    return __f


def neg_list(ls: Iterable[float]) -> Iterable[float]:
    return map(lambda x: -x)(ls)


def zip_with(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def __f(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        assert len(ls1) == len(ls2), "zip_with only works for equal size lists"  # type: ignore
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return __f


def add_lists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zip_with(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    def __f(ls):
        ans = start
        for item in ls:
            ans = fn(item, ans)
        return ans

    return __f


def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1)(ls)
