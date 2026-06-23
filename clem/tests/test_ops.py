import math

import clem
import numpy as np


def test_exp_log(approx):
    t = clem.tensor(10000.0)
    assert approx(clem.log(t).shape, ())


def test_positional_encoding_div_term(approx):
    d_model = 128
    i = clem.arange(0, d_model, 2)
    div_term = clem.exp(-i / d_model * clem.log(clem.tensor(10000.0)))
    ref = np.exp(-np.arange(0, d_model, 2) / d_model * np.log(10000.0))
    assert approx(list(div_term.shape), [d_model // 2])
    for got, expected in zip(list(div_term), ref):
        assert approx(got, float(expected))


def test_pos_broadcast_shape():
    max_seq_len = 256
    d_model = 128
    pos = clem.arange(max_seq_len, dtype="float32").reshape(max_seq_len, 1)
    div_term = clem.exp(-clem.arange(0, d_model, 2) / d_model * clem.log(clem.tensor(10000.0)))
    out = pos * div_term
    assert out.shape == (max_seq_len, d_model // 2)


def test_add_mul_scalar():
    t = clem.tensor([1.0, 2.0, 3.0])
    out = t * 2.0
    assert out.shape == t.shape
