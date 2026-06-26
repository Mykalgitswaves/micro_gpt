import clem


def test_import():
    assert clem is not None


def test_tensor_from_nested_list():
    t = clem.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)


def test_tensor_scalar():
    t = clem.tensor(3.14)
    assert t.shape == ()


def test_arange():
    t = clem.arange(0, 5, 2)
    assert t.shape == (3,)


def test_zeros():
    t = clem.zeros(2, 3)
    assert t.shape == (2, 3)


def test_randn_seeded():
    a = clem.randn(2, 2, seed=42)
    b = clem.randn(2, 2, seed=42)
    assert a.shape == b.shape


def test_reshape():
    t = clem.tensor([[1.0, 2.0], [3.0, 4.0]]).reshape(4)
    assert t.shape == (4,)
