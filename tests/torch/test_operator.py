import pytest

from hypothesis import given
from hypothesis.strategies import floats, integers

from toydl.torch.operator import mul

EPS = 1e-8
small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)


@pytest.mark.torch_operator
@given(small_floats, small_floats)
def test_same_as_python(x, y):
    """Check that the main operators all return the same value of the python version"""
    assert abs(mul(x, y) - x * y) < EPS
