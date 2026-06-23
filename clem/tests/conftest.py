import pytest

pytest.importorskip("clem")

RTOL = 1e-5
ATOL = 1e-6


@pytest.fixture
def approx():
    def _approx(a, b):
        import math

        if hasattr(a, "__iter__") and hasattr(b, "__iter__"):
            return all(math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL) for x, y in zip(a, b))
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)

    return _approx
