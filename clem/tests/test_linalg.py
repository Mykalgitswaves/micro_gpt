import clem
import numpy as np


def test_matmul_2d(approx):
    a = clem.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = clem.tensor([[2.0, 0.0], [1.0, 2.0]])
    out = a.matmul(b)
    ref = np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[2.0, 0.0], [1.0, 2.0]])
    assert out.shape == (2, 2)
    for row_out, row_ref in zip(out, ref):
        for got, expected in zip(row_out, row_ref):
            assert approx(got, expected)


def test_batched_matmul_smoke():
    a = clem.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    b = clem.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    out = a.matmul(b)
    assert out.shape == (1, 2, 2)


def test_transpose():
    t = clem.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = t.transpose(0, 1)
    assert out.shape == (2, 2)
