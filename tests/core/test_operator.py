import math

import pytest

from hypothesis import given

from toydl.core.operator import (
    add,
    eq,
    id_,
    inv,
    inv_back,
    log_back,
    lt,
    max_,
    mul,
    neg,
    relu,
    relu_back,
    sigmoid,
)

from .strategies import assert_close, small_floats


@pytest.mark.operator
@given(small_floats, small_floats)
def test_same_as_python(x, y):
    """Check that the main operators all return the same value of the python version"""
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max_(x, y), x if x > y else y)
    if x != 0.0 and not math.isinf(1.0 / x):
        assert_close(inv(x), 1.0 / x)


@pytest.mark.operator
@given(small_floats)
def test_relu(a):
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.operator
@given(small_floats, small_floats)
def test_relu_back(a, b):
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.operator
@given(small_floats)
def test_id(a):
    assert id_(a) == a


@pytest.mark.operator
@given(small_floats)
def test_lt(a):
    """Check that a - 1.0 is always less than a"""
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.operator
@given(small_floats)
def test_max(a):
    assert max_(a - 1.0, a) == a
    assert max_(a, a - 1.0) == a
    assert max_(a + 1.0, a) == a + 1.0
    assert max_(a, a + 1.0) == a + 1.0


@pytest.mark.operator
@given(small_floats)
def test_eq(a):
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.operator
@given(small_floats)
def test_sigmoid(a):
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as negative sigmoid
    * It crosses 0 at 0.5
    * it is  strictly increasing.
    """
    ans = sigmoid(a)
    assert 0 <= ans <= 1
    assert_close(1 - ans, sigmoid(-a))
    assert (a >= 0 and ans >= 0.5) or (a <= 0 and ans <= 0.5)
    assert sigmoid(a + 0.1) >= ans


@pytest.mark.operator
@given(small_floats, small_floats, small_floats)
def test_transitive(a, b, c):
    """Test the transitive property of less-than (a < b and b < c implies a < c)"""
    if (a < b) and (b < c):
        assert a < c


@pytest.mark.operator
@given(small_floats, small_floats)
def test_symmetric(a, b):
    """
    Write a test that ensures that mul is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    assert_close(mul(a, b), mul(b, a))


@pytest.mark.operator
@given(small_floats, small_floats, small_floats)
def test_distribute(x, y, z):
    r"""
    Write a test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    assert_close(mul(z, (x + y)), mul(z, x) + mul(z, y))


@pytest.mark.operator
@given(small_floats, small_floats)
def test_backs(a, b):
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
