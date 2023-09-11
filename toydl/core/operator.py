import math


def mul(x, y):
    """:math:`f(x, y) = x * y`"""
    return x * y


def mul_back(x, y, d):
    return d * y, d * x


def id(x):
    """:math:`f(x) = x`"""
    return x


def add(x, y):
    """:math:`f(x, y) = x + y`"""
    return x + y


def neg(x):
    """:math:`f(x) = -x`"""
    return -x


def neg_back(d):
    """:math:`f(x) = -x`"""
    return -d


def lt(x, y):
    """:math:`f(x) =` 1.0 if x is less than y else 0.0"""
    return float(x < y)


def eq(x, y):
    """:math:`f(x) =` 1.0 if x is equal to y else 0.0"""
    # Note: if x or y is float type, use `np.isclose` is better
    # return float(x == y)
    return abs(x - y) <= 1e-8


def max(x, y):
    """:math:`f(x) =` x if x is greater than y else y"""
    return x if x > y else y


def is_close(x, y):
    """:math:`f(x) = |x - y| < 1e-2` """
    return abs(x - y) < 1e-2


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    # IF-ELSE is used to make sure the `x` in exp(x) is always negative to avoid exp(x) overflow
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def sigmoid_back(x, d):
    r"""
    If :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` compute d :math:`d \times f'(x)`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Args:
        x (float): input
        d (float): derivative

    Returns:
        float : sigmoid value
    """
    return d * sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    return x if x > 0 else 0


def log(x):
    """:math:`f(x) = log(x)`"""
    return math.log(x)


def exp(x):
    """:math:`f(x) = e^{x}`"""
    return math.exp(x)


def exp_back(x, d):
    r"""If :math:`f = exp` compute d :math:`d \times f'(x)`"""
    return d * exp(x)


def log_back(x, d):
    r"""If :math:`f = log` as above, compute d :math:`d \times f'(x)`"""
    return d / (x)


def inv(x):
    """:math:`f(x) = 1/x`"""
    return 1 / x


def inv_back(x, d):
    r"""If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`"""
    return -d / x ** 2


def relu_back(x, d):
    r"""If :math:`f = relu` compute d :math:`d \times f'(x)`"""
    return d if x > 0 else 0


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    def __f(items):
        return [fn(item) for item in items]
    return __f


def neg_list(ls):
    """Use :func:`map` and :func:`neg` to negate each element in `ls`"""
    return map(lambda x: -x)(ls)


def zip_with(fn):
    """
    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """
    def __f(ls1, ls2):
        assert len(ls1) == len(ls2), "zipWith only works for equal size lists"
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return __f


def add_lists(ls1, ls2):
    """Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"""
    return zip_with(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def __f(ls):
        ans = start
        for item in ls:
            ans = fn(item, ans)
        return ans
    return __f


def sum(ls):
    """Sum up a list using :func:`reduce` and :func:`add`."""
    return reduce(add, 0)(ls)


def prod(ls):
    """Product of a list using :func:`reduce` and :func:`mul`."""
    return reduce(mul, 1)(ls)
