def mul(x, y):
    """:math:`f(x, y) = x * y`"""
    return x * y


def mul_back(x, y, d):
    return d * y, d * x
