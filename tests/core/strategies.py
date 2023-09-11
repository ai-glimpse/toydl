from hypothesis import settings
from hypothesis.strategies import floats

import toydl

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_floats = floats(min_value=-100, max_value=100, allow_nan=False)


def assert_close(a, b):
    assert toydl.core.operator.is_close(a, b), "Failure x=%f y=%f" % (a, b)
