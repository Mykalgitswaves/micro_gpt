import clem


def test_softmax_rows_sum_to_one(approx):
    t = clem.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    out = t.softmax(-1)
    for row in out:
        assert approx(sum(row), 1.0)


def test_layer_norm(approx):
    t = clem.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = clem.layer_norm_fn(t, [2])
    assert out.shape == (2, 2)
    flat = [out[i, j] for i in range(2) for j in range(2)]
    mean = sum(flat) / len(flat)
    assert approx(mean, 0.0,)


def test_gelu_at_zero(approx):
    t = clem.tensor([0.0])
    out = clem.gelu_fn(t)
    assert approx(out[0], 0.0)


def test_dropout_seeded():
    t = clem.tensor([1.0, 1.0, 1.0, 1.0])
    a = clem.dropout(t, p=0.5, seed=123)
    b = clem.dropout(t, p=0.5, seed=123)
    for x, y in zip(a, b):
        assert x == y
